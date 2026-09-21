# silent-miscalibration

Calibrated uncertainty for species distribution models under occurrence-data contamination.

This repository implements a calibration analysis for ensemble prediction intervals in species distribution models (SDMs), and applies split conformal prediction with leave-one-basin-out (LOBO) spatial calibration folds to restore near-nominal interval coverage under realistic data contamination. The repository accompanies the manuscript *Ensemble uncertainty fails under spatial bias in species distribution models — split conformal prediction restores calibrated coverage.*

The code regenerates per-replicate ensemble surfaces from the factorial benchmark of the companion methods paper, computes empirical coverage of contaminated-ensemble intervals against deterministic reference predictions, applies LOBO conformal correction, and produces all figures reported in the manuscript.

## Headline results

Across an eight-entity panel of crayfish SDMs fitted under three predictor tracks, two algorithms (Random Forest and XGBoost), and three contamination levels (3%, 10%, 20% low-accuracy occurrence records):

- Empirical coverage of contaminated-ensemble 95% prediction intervals against the contamination-free reference drops to as low as **0.49**, with monotonic decline across contamination levels.
- LOBO split conformal calibration restores empirical coverage to a panel-wide mean of **0.96** (138 of 144 cells at coverage ≥ 0.93).
- Width inflation cost ranges from **1.13× to 1.68×** across the panel, scaling with miscalibration severity.

Full numerical results are in `figures/panel_summary.csv` and `figures/panel_conformal.csv`.

## Repository layout

    trustworthy-sdm/
    ├── pyproject.toml
    ├── README.md
    ├── LICENSE
    ├── src/trustworthy_sdm/
    │   ├── io.py                       (loaders for the panel and benchmark surfaces)
    │   ├── regen.py                    (per-replicate surface regeneration)
    │   ├── analysis.py                 (coverage, asymmetry, panel-level evaluation)
    │   ├── conformal.py                (split conformal calibration with LOBO folds)
    │   └── cli.py                      (command-line entry points)
    ├── notebooks/
    │   ├── 02_full_panel.py            (uncorrected coverage analysis: Figures 2, 5, 6, S1)
    │   └── 03_conformal_calibration.py (conformal correction analysis: Figures 1, 3, 4)
    ├── slurm/
    │   ├── pilot_torrentium.sbatch     (single-entity pilot)
    │   └── full_panel_array.sbatch     (144-cell array job)
    ├── tests/
    └── scripts/
        └── generate_basin_lookups.py   (per-entity basin_id mapping)

## Installation

The package depends on the upstream `sdm-robustness` codebase (the companion paper's repository), pinned to a specific commit for reproducibility.

Clone both repositories side by side:

    git clone https://github.com/KristianMiok/trustworthy-sdm.git
    git clone https://github.com/KristianMiok/sdm-robustness.git

Then install in a virtual environment:

    cd trustworthy-sdm
    python3.12 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -e ../sdm-robustness
    pip install -e ".[dev]"

Verify the installation:

    python -c "from sdm_robustness.pipeline.core import fit_cv_cell; print('upstream OK')"
    trustworthy-sdm-inspect --help

Tested on Python 3.12 and 3.14, on macOS and Linux. The full pipeline runs in approximately ten minutes on a 32-core HPC partition; per-cell wall time is approximately two minutes for surface regeneration and one second for conformal evaluation.

## Reproducing the paper

Three steps reproduce every figure and tabulated value in the manuscript.

### 1. Regenerate the ensemble surfaces

The companion paper saves only aggregate metrics. Per-replicate predicted-suitability surfaces are required for the calibration analysis and are regenerated here from the companion paper's frozen seeds.

A single command regenerates all 4,320 surfaces (144 cells × 30 replicates):

    trustworthy-sdm-regenerate --cells full

On an HPC cluster, the equivalent Slurm array job:

    sbatch slurm/full_panel_array.sbatch

Output is written to `data/replicate_surfaces/`.

### 2. Run the analyses

    python notebooks/02_full_panel.py
    python notebooks/03_conformal_calibration.py

Output: cell-level summary tables in `figures/panel_summary.csv` and `figures/panel_conformal.csv`, and all figures referenced in the manuscript.

### 3. Verify with tests

    pytest tests/

All tests should pass on a fresh checkout with regenerated surfaces. Tests verify the conformal mathematics, the panel iteration, and end-to-end coverage on a held-out cell.

## Data dependencies

The analysis depends on three external resources:

- **The master occurrence database**, available through the World of Crayfish® platform (https://world.crayfish.ro/) and through Mendeley Data.
- **The Hydrography90m and GeoFRESH platforms** for network-aware predictor extraction.
- **The companion paper's panel definitions and deterministic benchmark surfaces**, distributed through that paper's archive.

For convenience during development, the `scripts/generate_basin_lookups.py` utility can pre-compute the per-entity basin lookups required for LOBO calibration. The lookup parquet files are small (approximately 1 MB total) and can be cached locally to avoid repeated calls to the upstream data-preparation pipeline.

## Citation

If you use this code, please cite the manuscript:

> *Ensemble uncertainty fails under spatial bias in species distribution models — split conformal prediction restores calibrated coverage.* In preparation.

## License

MIT. See `LICENSE`.
