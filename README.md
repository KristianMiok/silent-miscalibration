# silent-miscalibration

Code and analyses for the manuscript *Silent miscalibration of ensemble species distribution models under noisy occurrence data, and an adaptive conformal correction* (Environmental Data Science, EDS-2026-0083, in revision).

Occurrence records of lower positional accuracy are mixed into the training data of crayfish species distribution models, and the prediction intervals of the resulting ensembles are scored against a model fitted to high-accuracy records only (the clean benchmark). The repository measures how often those intervals miss the benchmark while ordinary cross-validation still looks fine, and applies split conformal calibration with leave-one-basin-out (LOBO) folds as a correction.

Two ensemble protocols:

- **Protocol A**: 30 seeded Random Forest or XGBoost fits per contaminated condition; interval = 2.5-97.5 percentiles over replicates; benchmark = the companion study's deterministic clean fit.
- **Protocol B**: GLM, GAM, Random Forest and XGBoost, each fitted once (in the revision, members calibrated with a fixed Platt map); interval = 2.5-97.5 percentiles of the four member predictions, just inside their min-max envelope; benchmark = mean of the four clean members.

Panel: eight crayfish units x three predictor tracks (local, upstream, combined) x three contamination levels (3, 10 and 20 % low-accuracy records).

## State of the repository

- Tag `eds-submitted` (f2d80c3, 10 Jul 2026): the submitted analysis.
- `main`: the revision. Revision analyses are `scripts/rev_eds_*.py`, their aggregate outputs `reports/rev_eds/`, the revision figures `figures/rev_eds/`.

## Headline results (revision)

- Protocol B, combined track: coverage of the clean benchmark 0.985 / 0.933 / 0.881 at L3 / L10 / L20, against 0.997 / 0.996 / 0.992 for matched-perturbation null refits (same training size, replaced records duplicated from high-accuracy ones). 0 / 3 / 11 of 24 configurations are silently miscovered: they pass an out-of-sample AUC >= 0.70 gate after contamination and cover the benchmark below 0.90.
- Protocol A: the 30-replicate interval covers the clean benchmark at a median of 0.90 even without contamination. With the miscoverage threshold set to min(0.90, 0.95 x null coverage) per configuration, 11 / 30 / 28 of 40 competent configurations are silently miscovered.
- LOBO split conformal calibration on calibrated Protocol B raises mean coverage from 0.914 to 0.969 (range 0.898-1.000); where the interval changes, its median width grows 1.35x.

## Installation

The analysis depends on the companion study's code, `sdm-robustness`, pinned to tag `v1.0` (8bbfb1d). Clone both side by side:

    git clone https://github.com/KristianMiok/silent-miscalibration.git
    git clone --branch v1.0 https://github.com/KristianMiok/sdm-robustness.git sdm-robustness-v1.0
    cd silent-miscalibration
    uv sync --frozen
    uv pip install --no-deps -e ../sdm-robustness-v1.0

## Data dependencies

- **Master occurrence table** (World of Crayfish, https://world.crayfish.ro/, and Mendeley Data): `combined_data_true_master.csv`, 115,191 records, md5 `0670eba9ae3098cb6c816493d8b13f79`. The revision scripts look for it at `../sdm-robustness/data/raw/` or at the path in `TS_MASTER_CSV`; `sdm-robustness` reads it through its own configuration. Network-aware predictors come from Hydrography90m and GeoFRESH through that table.
- **Companion study outputs** (`sdm-robustness` v1.0): Grid B results and deterministic benchmark surfaces under `data/results/`.
- **Record-level surfaces** (not tracked): Protocol A in `data/replicate_surfaces/` (`trustworthy-sdm-regenerate --cells full`, or `slurm/full_panel_array.sbatch`), Protocol B in `data/replicate_surfaces_protocol_b/` (`scripts/run_protocol_b.py`), revision surfaces in `data/rev_eds/`.
- **Odonata replicate**: the `filter_bias` repository side by side (`../filter_bias`; `run_odonata_overpred.py` at 458cf94, `reports/odonata_de_annotated.csv` md5 `5cd89ceff7078d62d4a489a2267b2a5f`).

## Reproducing the revision analyses

    scripts/reproduce_rev_eds.sh

runs every revision script in dependency order, logs each step, and checks with `git status` that every committed output in `reports/rev_eds/` and `figures/rev_eds/` is reproduced byte for byte. Record-level surfaces under `data/rev_eds/` are reused when present (about 7 minutes in total); remove that directory to regenerate them (several hours; the Protocol A null and agreement refits take about 70 minutes each on a 14-core laptop).

## Computing environments

Protocol B and all revision analyses run in the environment of `uv.lock` (Python 3.14.3, numpy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, xgboost 3.2.0, pygam 0.12.0); in it, the Protocol B gate refits the submitted members bit for bit.

The submitted Protocol A surfaces were generated in May 2026 on the VEGA cluster (IZUM). Regenerated here with the same code, they agree per surface at r = 0.967-1.000, and their coverage of the benchmark agrees within 0.010 in all 120 competent cells (median 0.001; `reports/rev_eds/protocol_a_local_agreement.csv`). Protocol A null refits are generated here; the worked case regenerates both arms here.

## Revision analyses

| script (`scripts/`) | purpose |
|---|---|
| `rev_eds_item1_protocol_a_oos.py`, `rev_eds_item1_protocol_a_join.py` | Protocol A out-of-sample (basin-blocked 5-fold CV) competence gate; performance against coverage |
| `rev_eds_gate_protocol_b_repro.py` | reproduction gate: Protocol B members refitted with v1.0 must match the submitted surfaces |
| `rev_eds_protocol_b_cv.py`, `rev_eds_item1_protocol_b_join.py` | Protocol B basin-blocked CV harness (out-of-fold predictions, training sites); performance against coverage |
| `rev_eds_diag_upstream_missingness.py` | whether the upstream-track signal is carried by missingness |
| `rev_eds_memorization_control_b.py` | coverage and directional gradient on never-trained sites |
| `rev_eds_item3_null_directional.py` | matched-perturbation null for the directional analysis (Protocol B) |
| `rev_eds_item2_calibration.py`, `rev_eds_item2b_fixed_map_calibration.py` | member calibration, per cell and with a fixed Platt map fitted on clean out-of-fold predictions |
| `rev_eds_item4_composition.py`, `rev_eds_item4b_metadata.py` | what distinguishes the low-accuracy records |
| `rev_eds_worked_case_null.py` | worked case (*P. leniusculus*): contaminated and null replicates regenerated locally |
| `rev_eds_protocol_b_calibrated.py` | full Protocol B panel and conformal correction on calibrated members |
| `rev_eds_item6_tables.py` | calibration audit, reviewer 3 points 13 and 14, supplement tables |
| `rev_eds_null_coverage_flips.py` | null-refit coverage (Figure 1 reference); worked-case flips at tau = 0.4 / 0.5 / 0.6 |
| `rev_eds_check_interval.py` | percentile interval against the min-max envelope |
| `rev_eds_odonata_null.py` | Odonata cross-taxon replicate with a matched null and a held-out domain |
| `rev_eds_worked_case_null_coverage.py`, `rev_eds_protocol_a_null_panel.py`, `rev_eds_protocol_a_local_agreement.py` | Protocol A null coverage, null-adjusted silent counts, local-vs-submitted agreement |
| `rev_eds_figures.py` | all revision figures |

Notes: the label `null` in `null_coverage.csv` (column `source`) and `odonata_null_reps.csv` (column `arm`) is read as missing by pandas' default NA parsing; read them with `keep_default_na=False, na_values=[""]`. `figures/` is git-ignored except the tracked revision figures in `figures/rev_eds/`.

## Citation

> *Silent miscalibration of ensemble species distribution models under noisy occurrence data, and an adaptive conformal correction.* Environmental Data Science, in revision (EDS-2026-0083).

## License

MIT. See `LICENSE`.
