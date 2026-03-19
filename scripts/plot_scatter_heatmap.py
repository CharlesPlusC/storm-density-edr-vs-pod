#!/usr/bin/env python3
"""Reproduce Figure 2 (density_scatter_heatmap) and Table 3 from NPZ files."""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ_DIR = os.path.join(REPO, 'data', 'npz')
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


def log_metrics(truth, model):
    """Log-normal accuracy metrics: beta, sigma, SD%, r, r^2, N."""
    truth = np.asarray(truth, dtype=float)
    model = np.asarray(model, dtype=float)
    valid = np.isfinite(truth) & np.isfinite(model) & (truth > 0) & (model > 0)
    t, m = truth[valid], model[valid]
    N = len(t)
    if N < 2:
        return {k: np.nan for k in ['beta', 'sigma', 'SD_pct', 'r', 'r_squared', 'N']}
    ln_ratio = np.log(m / t)
    beta = np.mean(ln_ratio)
    sigma = np.std(ln_ratio, ddof=1)
    SD_pct = 100.0 * (np.exp(sigma) - 1.0)
    r = np.corrcoef(t, m)[0, 1]
    return {'beta': beta, 'sigma': sigma, 'SD_pct': SD_pct, 'r': r,
            'r_squared': r**2 if np.isfinite(r) else np.nan, 'N': N}


def debias_arr(model, truth):
    """Remove median log-bias: rho_debiased = rho * exp(-median(ln(rho/truth)))."""
    m = np.asarray(model, dtype=float)
    t = np.asarray(truth, dtype=float)
    valid = np.isfinite(m) & np.isfinite(t) & (m > 0) & (t > 0)
    if valid.sum() < 5:
        return m.copy()
    beta = np.median(np.log(m[valid] / t[valid]))
    return m * np.exp(-beta)


def rms_pct(truth, model):
    """RMS percentage error, excluding outliers beyond 200%."""
    v = np.isfinite(truth) & np.isfinite(model) & (truth > 0) & (model > 0)
    if v.sum() < 10:
        return np.nan
    pct = (model[v] - truth[v]) / truth[v] * 100.0
    clip = np.abs(pct) < 200
    return np.sqrt(np.mean(pct[clip]**2))


def parse_sat_name(fname):
    """Extract satellite name and date from NPZ filename."""
    if fname.startswith('GRACE-FO-A_'):
        return 'GRACE-FO-A', fname[len('GRACE-FO-A_'):-4]
    parts = fname[:-4].split('_', 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return None, None


def drag_acc(rho, sat_name):
    """a_drag = 0.5 * rho * v^2 * Cd * A / m"""
    p = SAT_PARAMS[sat_name]
    return 0.5 * rho * p['v']**2 * p['Cd'] * p['area'] / p['mass']


def valid_mask(*arrays):
    """Return boolean mask where all arrays are finite and positive."""
    mask = np.ones(min(len(a) for a in arrays), dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a[:len(mask)]) & (a[:len(mask)] > 0)
    return mask


def load_all_npz():
    labels = ['3orb_matched', '2orb_matched', '1orb_matched',
              '1orb_native', 'suborb_best']
    pairs = {}
    for label in labels:
        for method in ['POD', 'EDR']:
            pairs[(label, method)] = {'truth': [], 'model': []}

    perorbit_rms = {m: {'drag': [], 'rms_pct': []} for m in ['POD', 'EDR']}

    npz_files = sorted(f for f in os.listdir(NPZ_DIR) if f.endswith('.npz'))
    print(f"Found {len(npz_files)} NPZ files")

    for fname in npz_files:
        sat, date_str = parse_sat_name(fname)
        if sat not in SAT_PARAMS:
            continue

        d = np.load(os.path.join(NPZ_DIR, fname), allow_pickle=True)

        for N, label in [(3, '3orb_matched'), (2, '2orb_matched'), (1, '1orb_matched')]:
            edr_key, truth_edr_key = f'edr_{N}orb', f'edr_truth_{N}orb'
            pod_key, truth_pod_key = f'pod_{N}orb', f'pod_truth_{N}orb'

            if edr_key not in d or pod_key not in d:
                continue

            edr_N = np.asarray(d[edr_key], dtype=float)
            truth_edr_N = np.asarray(d[truth_edr_key], dtype=float)
            pod_N = np.asarray(d[pod_key], dtype=float)
            truth_pod_N = np.asarray(d[truth_pod_key], dtype=float)

            n = min(len(edr_N), len(truth_edr_N), len(pod_N), len(truth_pod_N))
            edr_db = debias_arr(edr_N[:n], truth_edr_N[:n])
            pod_db = debias_arr(pod_N[:n], truth_pod_N[:n])

            v = valid_mask(edr_db[:n], truth_edr_N[:n], pod_db[:n], truth_pod_N[:n])

            pairs[(label, 'EDR')]['truth'].extend(truth_edr_N[:n][v].tolist())
            pairs[(label, 'EDR')]['model'].extend(edr_db[:n][v].tolist())
            pairs[(label, 'POD')]['truth'].extend(truth_pod_N[:n][v].tolist())
            pairs[(label, 'POD')]['model'].extend(pod_db[:n][v].tolist())

        td_native = np.asarray(d['td_on_sp3'], dtype=float)

        if 'edr_step_sp3' in d and 'pod_step_sp3' in d:
            edr_step = np.asarray(d['edr_step_sp3'], dtype=float)
            pod_step = np.asarray(d['pod_step_sp3'], dtype=float)
            n = min(len(edr_step), len(pod_step), len(td_native))
            v = valid_mask(edr_step[:n], pod_step[:n], td_native[:n])
            for method, arr in [('EDR', edr_step), ('POD', pod_step)]:
                pairs[('1orb_native', method)]['truth'].extend(td_native[:n][v].tolist())
                pairs[('1orb_native', method)]['model'].extend(arr[:n][v].tolist())

        has_edr_sub = 'edr_suborb' in d and np.asarray(d['edr_suborb']).size > 0
        has_pod_sub = 'pod_suborb' in d and np.asarray(d['pod_suborb']).size > 0
        if has_edr_sub and has_pod_sub:
            edr_sub = np.asarray(d['edr_suborb'], dtype=float)
            pod_sub = np.asarray(d['pod_suborb'], dtype=float)
            n = min(len(edr_sub), len(pod_sub), len(td_native))
            v = valid_mask(edr_sub[:n], pod_sub[:n], td_native[:n])
            for method, arr in [('EDR', edr_sub), ('POD', pod_sub)]:
                pairs[('suborb_best', method)]['truth'].extend(td_native[:n][v].tolist())
                pairs[('suborb_best', method)]['model'].extend(arr[:n][v].tolist())

            perigees = np.asarray(d['perigees_sp3'], dtype=int)
            for k in range(len(perigees) - 1):
                i0, i1 = perigees[k], perigees[k + 1]
                if i1 > n:
                    break
                orb_v = v[i0:i1]
                if orb_v.sum() < 10:
                    continue
                orb_td = td_native[i0:i1][orb_v]
                orb_drag = np.mean([drag_acc(rho, sat) for rho in orb_td])
                for method, sub_arr in [('EDR', edr_sub), ('POD', pod_sub)]:
                    orb_model = sub_arr[i0:i1][orb_v]
                    orb_pct = np.abs((orb_model - orb_td) / orb_td) * 100.0
                    perorbit_rms[method]['drag'].append(orb_drag)
                    perorbit_rms[method]['rms_pct'].append(np.sqrt(np.mean(orb_pct**2)))

    for key in pairs:
        pairs[key]['truth'] = np.array(pairs[key]['truth'])
        pairs[key]['model'] = np.array(pairs[key]['model'])

    for m in perorbit_rms:
        perorbit_rms[m]['drag'] = np.array(perorbit_rms[m]['drag'])
        perorbit_rms[m]['rms_pct'] = np.array(perorbit_rms[m]['rms_pct'])

    return pairs, perorbit_rms


def plot_scatter_heatmap(pairs):
    LOG_LO, LOG_HI = -14.5, -10.5
    NBINS = 120

    def log_fmt(x, _):
        exp = int(round(x))
        return f'$10^{{{exp}}}$' if abs(x - exp) < 0.01 else ''

    row_configs = [
        ('3orb_matched',  '3-orbit eff (~270 min)\nvs matched TU Delft'),
        ('2orb_matched',  '2-orbit eff (~180 min)\nvs matched TU Delft'),
        ('1orb_matched',  '1-orbit eff (~90 min)\nvs matched TU Delft'),
        ('1orb_native',   '1-orbit eff\nvs native TU Delft (30 s)'),
        ('suborb_best',   'Sub-orbital (best arc)\nvs native TU Delft'),
    ]

    n_rows = len(row_configs)
    fig = plt.figure(figsize=(7.5, 16))
    gs = fig.add_gridspec(n_rows, 3, width_ratios=[1, 1, 0.04],
                          wspace=0.30, hspace=0.35)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(n_rows)]
    cbar_ax = fig.add_subplot(gs[:, 2])

    panel_labels = 'abcdefghijklmnop'
    stats_rows = []

    for row_i, (label, row_title) in enumerate(row_configs):
        pod_t = pairs[(label, 'POD')]['truth']
        pod_m = pairs[(label, 'POD')]['model']
        edr_t = pairs[(label, 'EDR')]['truth']
        edr_m = pairs[(label, 'EDR')]['model']
        common_valid = (np.isfinite(pod_t) & np.isfinite(pod_m)
                        & np.isfinite(edr_t) & np.isfinite(edr_m)
                        & (pod_t > 0) & (pod_m > 0)
                        & (edr_t > 0) & (edr_m > 0))

        for col_i, method in enumerate(['POD', 'EDR']):
            ax = axes[row_i][col_i]
            panel = panel_labels[row_i * 2 + col_i]

            truth = pairs[(label, method)]['truth'][common_valid]
            model = pairs[(label, method)]['model'][common_valid]

            met = log_metrics(truth, model)
            r2 = met['r_squared']
            rms = rms_pct(truth, model)

            vis = ((truth >= 10**LOG_LO) & (truth <= 10**LOG_HI)
                   & (model >= 10**LOG_LO) & (model <= 10**LOG_HI))
            truth_v = truth[vis]
            model_v = model[vis]

            if len(truth_v) < 10:
                ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                        ha='center', va='center')
                continue

            stats_rows.append({
                'resolution': label, 'method': method,
                'r2': r2, 'sigma': met['sigma'], 'SD_pct': met['SD_pct'],
                'RMS_pct': rms, 'N': met['N'],
            })

            log_t = np.log10(truth_v)
            log_m = np.log10(model_v)

            ax.set_facecolor('#e8e0d4')
            bins = np.linspace(LOG_LO, LOG_HI, NBINS + 1)
            ax.hist2d(log_t, log_m, bins=bins, cmap='gist_heat', norm=LogNorm(vmin=1))
            ax.plot([LOG_LO, LOG_HI], [LOG_LO, LOG_HI], 'w--', lw=1.0, alpha=0.8)
            ax.set_xlim(LOG_LO, LOG_HI)
            ax.set_ylim(LOG_LO, LOG_HI)
            ax.set_aspect('equal')

            ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_fmt))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_fmt))
            ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
            ax.yaxis.set_major_locator(mticker.MultipleLocator(1))

            ax.text(0.05, 0.95, f"({panel})", transform=ax.transAxes, fontsize=9,
                    va='top', ha='left', color='white', fontweight='bold')

            if row_i == 0:
                ax.set_title(method, fontsize=14, fontweight='bold')
            if col_i == 0:
                ax.set_ylabel('Method [kg/m$^3$]', fontsize=9)
            if row_i == n_rows - 1:
                ax.set_xlabel('Accelerometer [kg/m$^3$]', fontsize=9)
            if row_i < n_rows - 1:
                ax.set_xticklabels([])
            if col_i > 0:
                ax.set_yticklabels([])

        axes[row_i][1].text(1.08, 0.5, row_title, transform=axes[row_i][1].transAxes,
                             fontsize=8, va='center', ha='left', rotation=-90,
                             fontweight='bold')

    sm = plt.cm.ScalarMappable(cmap='gist_heat', norm=LogNorm(vmin=1, vmax=300))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label='Counts')

    for ext in ['png', 'svg']:
        fig.savefig(os.path.join(FIG_DIR, f'density_scatter_heatmap.{ext}'),
                    bbox_inches='tight', dpi=200 if ext == 'png' else None)
    plt.close(fig)
    print("Saved: figures/density_scatter_heatmap.png + .svg")

    return pd.DataFrame(stats_rows)


def format_heatmap_ax(ax, x_lo, x_hi, y_lo, y_hi, show_ylabel=True):
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(r'Drag Acceleration [m/s$^2$]', fontsize=10)
    if show_ylabel:
        ax.set_ylabel(r'$|\tilde{\rho}_{\mathrm{err}}|$ [%]', fontsize=10)
    else:
        ax.set_yticklabels([])

    def drag_fmt(val, _):
        exp = int(round(val))
        return f'$10^{{{exp}}}$' if abs(val - exp) < 0.01 else ''

    def err_fmt(val, _):
        lin = 10**val
        return f'{lin:.0f}' if lin >= 1 else f'{lin:.1f}'

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(drag_fmt))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(err_fmt))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))


def plot_rms_vs_drag_perorbit(perorbit_rms, num_bins=12):
    fig = plt.figure(figsize=(15, 4.5))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.04, 1, 0.04],
                          wspace=0.12, left=0.08)

    x_lo, x_hi = -8.5, -5.5
    y_lo = np.log10(5)
    y_hi = 2.5
    nx, ny = 35, 35
    x_edges = np.linspace(x_lo, x_hi, nx + 1)
    y_edges = np.linspace(y_lo, y_hi, ny + 1)

    ax_pod = fig.add_subplot(gs[0, 0])
    ax_edr = fig.add_subplot(gs[0, 1])
    cbar_ax = fig.add_subplot(gs[0, 2])
    ax_diff = fig.add_subplot(gs[0, 3])
    diff_cbar_ax = fig.add_subplot(gs[0, 4])

    hists = {}

    for col_i, (method, ax) in enumerate([('POD', ax_pod), ('EDR', ax_edr)]):
        drag = perorbit_rms[method]['drag']
        rms_arr = perorbit_rms[method]['rms_pct']
        mask = (drag > 0) & np.isfinite(rms_arr)
        x = np.log10(drag[mask])
        y = np.log10(rms_arr[mask].clip(min=0.5))

        H, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
        hists[method] = H
        H_masked = np.ma.masked_where(H == 0, H)
        cmap_heat = plt.cm.gist_heat.copy()
        ax.pcolormesh(x_edges, y_edges, H_masked.T, cmap=cmap_heat,
                      norm=LogNorm(vmin=0.5, vmax=max(H.max(), 1)),
                      rasterized=True)

        edges_rms = np.logspace(x_lo, x_hi, num_bins + 1)
        inds = np.clip(np.digitize(drag[mask], edges_rms) - 1, 0, num_bins - 1)
        centers = np.sqrt(edges_rms[:-1] * edges_rms[1:])
        median_rms = np.full(num_bins, np.nan)
        for i in range(num_bins):
            in_bin = rms_arr[mask][inds == i]
            if len(in_bin) >= 3:
                median_rms[i] = np.median(in_bin)
        v = np.isfinite(median_rms)
        ax.plot(np.log10(centers[v]), np.log10(median_rms[v]),
                color='white', marker='o', ms=4, lw=2.2, zorder=4, label='Median')

        ray = RAY_POD if method == 'POD' else RAY_EDR
        ax.plot(np.log10(ray[:, 0]), np.log10(ray[:, 1]),
                '--', color='lime', lw=2.0, zorder=4, label='Ray et al.')

        ax.set_title(method, fontsize=13, fontweight='bold')
        format_heatmap_ax(ax, x_lo, x_hi, y_lo, y_hi, show_ylabel=(col_i == 0))
        ax.legend(fontsize=7, loc='upper right',
                  fancybox=True, framealpha=0.85, edgecolor='black')
        print(f"  {method}: {mask.sum():,} orbits")

    vmax_h = max(h.max() for h in hists.values())
    sm = plt.cm.ScalarMappable(cmap='gist_heat',
                                norm=LogNorm(vmin=0.5, vmax=max(vmax_h, 1)))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label='Counts')

    diff = hists['POD'].astype(float) - hists['EDR'].astype(float)
    both_zero = (hists['POD'] == 0) & (hists['EDR'] == 0)
    diff_masked = np.ma.masked_where(both_zero, diff)
    nonzero = diff[~both_zero] if (~both_zero).any() else np.array([0])
    vmax_diff = max(np.percentile(np.abs(nonzero), 98), 1)
    im = ax_diff.pcolormesh(x_edges, y_edges, diff_masked.T,
                            cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff,
                            rasterized=True)
    ax_diff.set_title('POD $-$ EDR', fontsize=13, fontweight='bold')
    format_heatmap_ax(ax_diff, x_lo, x_hi, y_lo, y_hi, show_ylabel=False)
    fig.colorbar(im, cax=diff_cbar_ax, label=r'$\Delta$ Counts')

    for ext in ['png', 'svg']:
        fig.savefig(os.path.join(FIG_DIR, f'rms_vs_drag_perorbit.{ext}'),
                    bbox_inches='tight', dpi=200 if ext == 'png' else None,
                    transparent=True)
    plt.close(fig)
    print("Saved: figures/rms_vs_drag_perorbit.png + .svg")


def save_table3(stats_df):
    stats_df.to_csv(os.path.join(FIG_DIR, 'table3.csv'), index=False)

    with open(os.path.join(FIG_DIR, 'table3.txt'), 'w') as f:
        header = (f"{'Resolution':<20s} {'Method':<6s} {'SD%':>7s} {'r2':>7s} "
                  f"{'sigma':>7s} {'RMS%':>7s} {'N':>8s}")
        sep = '-' * len(header)
        for line in [header, sep]:
            print(line)
            f.write(line + '\n')
        for _, r in stats_df.iterrows():
            line = (f"{r['resolution']:<20s} {r['method']:<6s} "
                    f"{r['SD_pct']:7.1f} {r['r2']:7.3f} "
                    f"{r['sigma']:7.3f} {r['RMS_pct']:7.1f} {int(r['N']):8d}")
            print(line)
            f.write(line + '\n')

    print("\nSaved: figures/table3.csv + .txt")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    print("Loading NPZ data...")
    pairs, perorbit_rms = load_all_npz()
    for label in ['3orb_matched', '2orb_matched', '1orb_matched',
                  '1orb_native', 'suborb_best']:
        for method in ['POD', 'EDR']:
            n = len(pairs[(label, method)]['truth'])
            print(f"  {label:20s} {method}: {n:6d} pairs")

    print("\n=== Scatter Heatmap (Figure 2) ===")
    stats_df = plot_scatter_heatmap(pairs)

    print("\n=== Table 3 ===")
    save_table3(stats_df)

    print("\n=== Per-Orbit RMS Heatmap ===")
    plot_rms_vs_drag_perorbit(perorbit_rms)


if __name__ == '__main__':
    main()
