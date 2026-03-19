#!/usr/bin/env python3
"""Reproduce all paper figures and Table 3.  Usage: python scripts/reproduce_all.py"""

import subprocess
import sys
import os

SCRIPTS = [
    'plot_scatter_heatmap.py',
    'plot_rms_vs_drag.py',
    'plot_tradeoff.py',
]

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    failed = []
    for script in SCRIPTS:
        path = os.path.join(script_dir, script)
        print(f"\n{'='*60}")
        print(f"Running {script}...")
        print('='*60)
        ret = subprocess.call([sys.executable, path])
        if ret != 0:
            failed.append(script)
            print(f"WARNING: {script} exited with code {ret}")

    fig_dir = os.path.join(os.path.dirname(script_dir), 'figures')
    print(f"\n{'='*60}")
    print("DONE")
    print('='*60)
    print(f"\nFigures saved to {fig_dir}/:")
    for f in sorted(os.listdir(fig_dir)):
        print(f"  {f}")

    if failed:
        print(f"\nWARNING: {len(failed)} script(s) failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
