#!/usr/bin/env python3
"""Reproduce Figure 5 (tradeoff_drag_vs_regime) from tradeoff_pooled.csv."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO, 'figures')

WINDOWS = [2, 3, 5, 7, 10, 15, 20, 30, 45, 60, 75, 90, 120, 150, 180, 210, 240, 270]


def load_tradeoff():
    pooled = pd.read_csv(os.path.join(REPO, 'data', 'tradeoff_pooled.csv'))
    n_storms = pooled[['satellite', 'storm_date']].drop_duplicates().shape[0]
    print(f"Loaded {n_storms} storms, {len(pooled)} rows")

    pooled["r2_native"] = pooled["r_native"] ** 2
    pooled["r2_matched"] = pooled["r_matched"] ** 2
    pooled["sigma_native"] = np.log(1 + pooled["SD_pct_native"] / 100.0)
    return pooled


def plot_tradeoff(pooled):
    fig, all_axes = plt.subplots(2, 2, figsize=(9, 6), sharey='row', sharex=True)

    row_specs = [
        ("sigma_native", "sigma",
         r"$\sigma$" + "\n" + r"(log $\rho_{\mathrm{retrieved}}$ / $\rho_{\mathrm{ACC}}$)"),
        ("r2_native", "r2_matched", r"$r^2$"),
    ]

    w = np.array(WINDOWS)

    for col_i, method in enumerate(["POD", "EDR"]):
        mdf = pooled[pooled["method"] == method]
        for row_i, (nat_col, match_col, ylabel) in enumerate(row_specs):
            ax = all_axes[row_i][col_i]

            for metric, color, label in [
                (nat_col, '#2980b9', 'Native (0.1Hz)'),
                (match_col, '#e67e22', 'Matched'),
            ]:
                grp = mdf.groupby("window_min")[metric]
                med = grp.median().reindex(WINDOWS)
                std = grp.std().reindex(WINDOWS)
                ax.fill_between(w, (med - std).values, (med + std).values,
                                alpha=0.2, color=color)
                ax.plot(w, med.values, color=color, lw=2, label=label)

            ax.set_facecolor('#f5f0eb')
            ax.set_xscale("log")
            ax.set_xticks(WINDOWS)
            ax.set_xticklabels([str(v) for v in WINDOWS], fontsize=7, rotation=45)
            if col_i == 0:
                ax.set_ylabel(ylabel, fontsize=10)
            if row_i == 0:
                ax.set_title("POD-A" if method == "POD" else "EDR",
                             fontsize=13, fontweight="bold")
            if row_i == 1:
                ax.set_xlabel("Window / Arc length [minutes]", fontsize=10)
            if col_i == 0 and row_i == 0:
                ax.legend(fontsize=8, loc="best",
                          fancybox=True, framealpha=0.9, edgecolor='black')
            ax.grid(True, alpha=0.3)

    fig.tight_layout()

    for ext in ['png', 'svg']:
        fig.savefig(os.path.join(FIG_DIR, f'tradeoff_drag_vs_regime.{ext}'),
                    bbox_inches='tight', dpi=200 if ext == 'png' else None)
    plt.close(fig)
    print("Saved: figures/tradeoff_drag_vs_regime.png + .svg")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    print("Loading tradeoff data...")
    pooled = load_tradeoff()

    print("\n=== Figure 5: Accuracy-Resolution Trade-off ===")
    plot_tradeoff(pooled)


if __name__ == '__main__':
    main()
