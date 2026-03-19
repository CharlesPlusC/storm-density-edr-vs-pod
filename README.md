# Reproducibility Package

**Paper:** *Empirical Assessment of Storm-time Thermospheric Density Inversion Methods from LEO POD Data*

**Authors:** Charles Constant, Indigo Brownhall, Anasuya Aruliah, Marek Ziebart, Santosh Bhattarai

**Affiliation:** University College London

**Journal:** Earth and Space Science (AGU)

---

## Quick Start (Tier 1: ~30 seconds)

Reproduces all results figures and Table 3 from pre-computed intermediate data.

```bash
pip install -r requirements.txt
python scripts/reproduce_all.py
```

All outputs are saved to `figures/`.

## Figures and Scripts

| Paper Fig | Filename | Script | Input Data |
|-----------|----------|--------|------------|
| 1 | `f107_storm_timeline.png` | Pre-made (contextual) | -- |
| 2 | `density_scatter_heatmap.png` | `scripts/plot_scatter_heatmap.py` | `data/npz/*.npz` |
| 3 | `rms_vs_drag_orbit_effective.png` | `scripts/plot_rms_vs_drag.py` | `data/orbit_effective/*/` |
| 4 | `per-orbit-hist.png` | `scripts/plot_rms_vs_drag.py` | `data/orbit_effective/*/` |
| 5 | `tradeoff_drag_vs_regime.png` | `scripts/plot_tradeoff.py` | `data/tradeoff_pooled.csv` |
| Table 3 | `table3.csv` / `table3.txt` | `scripts/plot_scatter_heatmap.py` | `data/npz/*.npz` |
| -- | `rms_vs_drag_perorbit.png` | `scripts/plot_scatter_heatmap.py` | `data/npz/*.npz` |

## Data Description

### `data/npz/` (49 files, ~16 MB)
Per-storm NumPy archives containing density time series at multiple resolutions:
- `edr_1orb`, `edr_2orb`, `edr_3orb`: N-orbit effective densities (EDR method)
- `pod_1orb`, `pod_2orb`, `pod_3orb`: N-orbit effective densities (POD-A method)
- `edr_truth_1orb`, `edr_truth_2orb`, `edr_truth_3orb`: Matched accelerometer-derived truth (EDR perigees)
- `pod_truth_1orb`, `pod_truth_2orb`, `pod_truth_3orb`: Matched accelerometer-derived truth (POD perigees)
- `edr_step_sp3`, `pod_step_sp3`: 1-orbit step functions on SP3 30 s grid
- `td_on_sp3`: Accelerometer-derived truth interpolated to SP3 grid
- `edr_suborb`, `pod_suborb`: Sub-orbital (optimal arc) densities
- `perigees_sp3`, `perigees_pod`: Perigee indices on SP3 and POD grids
- `edr_best_arc`, `pod_best_win`: Optimal sub-orbital arc/window length (minutes)

### `data/orbit_effective/` (49 storm directories, ~760 KB)
Per-storm CSVs with 1-orbit effective density:
- `edr_effective.csv`: columns `time`, `edr_raw`, `edr_debiased`
- `pod_acc_effective.csv`: columns `time`, `acc_effective` (truth), `pod_raw`, `pod_debiased`

CHAMP data is pre-filtered to retain only orbits where all density values (EDR, POD-A, and accelerometer truth) are finite and positive, giving 1513 valid orbits across 22 storms.

### `data/tradeoff_pooled.csv` (~460 KB)
Accuracy-resolution sweep: sigma, r, SD% at each (storm, method, window length) combination.

### `data/tradeoff_optimal.csv` (~12 KB)
One row per (storm, method) with the optimal window that maximises native-resolution correlation.

### `data/hires_summary.csv` (~16 KB)
Per-storm summary with mean drag acceleration and optimal windows for both methods.

### `data/storms.csv`
List of all 49 storms (22 CHAMP + 27 GRACE-FO-A) with Kp category.

## Tier 2: Full Regeneration from Raw Data

The `src/` directory contains the complete analysis library used to generate the intermediate data from raw inputs. Regeneration requires:

- **Orekit** (Java-based astrodynamics library via Python wrapper)
- **SP3 precise ephemeris files** (ESA GNSS archives)
- **TU Delft accelerometer-derived densities** (thermosphere.tudelft.nl)
- **Space weather indices** (NOAA SWPC)
- **tqdm** (progress bars for batch processing)

See `scripts/compute_full_table3.py` and `scripts/run_all_storms.py` for the Tier 2 entry points. These require `src/` to be importable with an initialised Orekit VM.

### Satellite Parameters (Table 2 in paper)

| Parameter | CHAMP | GRACE-FO-A |
|-----------|-------|------------|
| Mass [kg] | 522.0 | 600.2 |
| Cross-section [m²] | 1.0 | 1.04 |
| C_D | 2.2 | 3.2 |

## Citation

If you use this software or data, please cite:

Constant, C., Brownhall, I., Aruliah, A., Ziebart, M., & Bhattarai, S. (2026). *Empirical Assessment of Storm-time Thermospheric Density Inversion Methods from LEO POD Data.* Earth and Space Science.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19110462.svg)](https://doi.org/10.5281/zenodo.19110462)

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
