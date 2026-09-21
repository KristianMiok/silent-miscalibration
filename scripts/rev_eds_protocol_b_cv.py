"""EDS-2026-0083 revision, items 1-2 (Protocol B): basin-blocked CV harness.

For every Protocol B cell (8 entities x 3 tracks x L0/L3/L10/L20):
  1. basin-blocked 5-fold cross-validation of the four consensus members
     (GLM, GAM, RF, XGBoost) and of the consensus (mean and median), mirroring
     the companion paper's fit_cv_cell at sdm-robustness v1.0: folds assigned
     by presence basin (assign_basin_folds, 5 splits, LOOO threshold 15); per
     fold a fresh 1:1 background draw from the accessible area (seed SEED+fold)
     split by the same basin map, unmapped basins to fold 0; metrics from
     compute_performance_metrics at threshold 0.5, averaged over folds;
  2. out-of-fold member predictions are kept (input for member calibration);
  3. the training sites of the submitted full-data fit are recorded (kept and
     contaminant presences, background), for the memorization control;
  4. the full-data members are refitted and compared with the submitted
     surfaces (reproduction check over every cell; exit code 1 on failure).

Members are fitted with scripts/run_protocol_b.py's own fit_one_algorithm, so
predictor sets, scaling and hyperparameters are those of the submission.

Outputs
  reports/rev_eds/item1_protocol_b_oos.csv    one row per cell x model
  reports/rev_eds/protocol_b_repro_all.csv    one row per cell x member
  data/rev_eds/protocol_b_cv/<cell>/{metrics.csv, oof.parquet,
      training_sites.parquet, repro.csv}
  data/rev_eds/protocol_b_cv/<entity_dir>__pool_sites.parquet
Finished cells are skipped unless --force.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
MASTER = Path(os.environ.get("TS_MASTER_CSV") or
              REPO.parent / "sdm-robustness/data/raw/combined_data_true_master.csv")
WORK = REPO / "data/rev_eds/protocol_b_cv"
OUT_OOS = REPO / "reports/rev_eds/item1_protocol_b_oos.csv"
OUT_REPRO = REPO / "reports/rev_eds/protocol_b_repro_all.csv"
N_SPLITS = 5
LOOO_THRESHOLD = 15
REPRO_TOL = 1e-6
AUC_GATE = 0.70


def load_runner():
    spec = importlib.util.spec_from_file_location("run_protocol_b", REPO / "scripts/run_protocol_b.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def training_presences(rpb, inputs, level):
    """Presence set exactly as consensus_surfaces_for_cell builds it, plus a role column.

    contaminate_presence_set keeps n - k benchmark records and appends k pool
    records, so the contaminants are the last k rows.
    """
    from sdm_robustness.pipeline.core import contaminate_presence_set
    bench = inputs.benchmark
    if level == 0:
        pres = bench.copy()
        k = 0
    else:
        pres = contaminate_presence_set(benchmark=bench, contamination_pool=inputs.contamination_pool,
                                        level_pct=level, seed=rpb.SEED).copy()
        k = int(round(len(bench) * level / 100.0))
    pres = pres.reset_index(drop=True)
    pres["role"] = ["kept"] * (len(pres) - k) + ["contaminant"] * k
    return pres


def cv_cell(rpb, inputs, pres, reduced_cols, full_cols, assign_basin_folds, perf):
    acc = inputs.accessible_area
    bench = inputs.benchmark
    fold_map = assign_basin_folds(pres["basin_id"], n_splits=N_SPLITS, looo_threshold=LOOO_THRESHOLD)
    pres = pres.copy()
    pres["fold"] = pres["basin_id"].astype(str).map(fold_map)
    n_neg = min(int(round(len(pres) * rpb.PA_RATIO)), len(acc))
    fold_rows, oof = [], []
    for fold in sorted(pres["fold"].dropna().unique().tolist()):
        p_tr = pres[pres["fold"] != fold]
        p_te = pres[pres["fold"] == fold]
        if p_tr.empty or p_te.empty:
            continue
        neg = acc.sample(n=n_neg, replace=False, random_state=rpb.SEED + int(fold)).copy()
        neg["fold"] = neg["basin_id"].astype(str).map(fold_map).fillna(0).astype(int)
        n_tr = neg[neg["fold"] != fold]
        n_te = neg[neg["fold"] == fold]
        y_tr = np.r_[np.ones(len(p_tr)), np.zeros(len(n_tr))].astype(int)
        y_te = np.r_[np.ones(len(p_te)), np.zeros(len(n_te))].astype(int)
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue
        preds = {}
        for algo in rpb.CONSENSUS_ORDER:
            cols = reduced_cols if algo in {"glm", "gam"} else full_cols
            med = bench[cols].median(numeric_only=True)
            x_tr = pd.concat([p_tr[cols].fillna(med), n_tr[cols].fillna(med)], axis=0)
            x_te = pd.concat([p_te[cols].fillna(med), n_te[cols].fillna(med)], axis=0)
            preds[algo] = np.clip(rpb.fit_one_algorithm(algo, x_tr, y_tr, x_te, rpb.SEED), 0, 1)
        members = np.vstack([preds[a] for a in rpb.CONSENSUS_ORDER])
        preds["consensus_mean"] = members.mean(axis=0)
        preds["consensus_median"] = np.median(members, axis=0)
        for model, score in preds.items():
            fold_rows.append({"fold": int(fold), "model": model, **perf(y_te, score, threshold=0.5)})
        te = pd.concat([p_te[["subc_id", "basin_id", "role"]],
                        n_te[["subc_id", "basin_id"]].assign(role="background")], ignore_index=True)
        te["fold"] = int(fold)
        te["y"] = y_te
        for algo in rpb.CONSENSUS_ORDER:
            te[algo] = preds[algo]
        oof.append(te)
    return pd.DataFrame(fold_rows), (pd.concat(oof, ignore_index=True) if oof else pd.DataFrame())


def full_fit_sites(rpb, inputs, pres):
    acc = inputs.accessible_area
    n_neg = min(int(round(len(pres) * rpb.PA_RATIO)), len(acc))
    neg = acc.sample(n=n_neg, replace=False, random_state=rpb.SEED)
    return pd.concat([pres[["subc_id", "basin_id", "role"]],
                      neg[["subc_id", "basin_id"]].assign(role="background")], ignore_index=True)


def repro_check(rpb, inputs, track, level, reduced_cols, full_cols, saved_dir):
    surf = rpb.consensus_surfaces_for_cell(inputs, track, level, reduced_cols, full_cols, rpb.SEED)
    rows = []
    for algo, s in surf.items():
        saved = pd.read_parquet(saved_dir / f"{rpb.ALGO_TO_REP[algo]}.parquet")
        sv = pd.Series(saved["predicted_probability"].values, index=saved["subc_id"].astype(str).values)
        nw = pd.Series(np.asarray(s.values, dtype=float), index=pd.Index(s.index).astype(str))
        common = sv.index.intersection(nw.index)
        diff = (nw.loc[common] - sv.loc[common]).abs()
        rows.append({"algorithm": algo, "n_saved": len(sv), "n_new": len(nw), "n_common": len(common),
                     "max_abs_diff": float(diff.max()) if len(common) else np.nan})
    return pd.DataFrame(rows)


def summarize():
    oos = pd.concat([pd.read_csv(p) for p in sorted(WORK.glob("*/metrics.csv"))], ignore_index=True)
    OUT_OOS.parent.mkdir(parents=True, exist_ok=True)
    oos.round(4).to_csv(OUT_OOS, index=False)
    n_cells = oos[["entity", "track", "level"]].drop_duplicates().shape[0]
    print(f"\n{n_cells} cells -> {OUT_OOS.relative_to(REPO)}")
    print(oos.pivot_table(index=["model", "track"], columns="level", values="auc", aggfunc="mean").round(3).to_string())
    clean = oos[(oos["level"] == 0) & (oos["model"] == "consensus_mean")]
    weak = clean[clean["auc"] < AUC_GATE][["entity", "track", "auc"]]
    print(f"\nclean consensus below the competence gate (AUC < {AUC_GATE}):")
    print(weak.round(3).to_string(index=False) if len(weak) else "none")
    reps = sorted(WORK.glob("*/repro.csv"))
    if not reps:
        return True
    rep = pd.concat([pd.read_csv(p) for p in reps], ignore_index=True)
    rep.to_csv(OUT_REPRO, index=False)
    bad = rep[(rep["max_abs_diff"] >= REPRO_TOL) | (rep["n_common"] != rep["n_saved"])
              | (rep["n_common"] != rep["n_new"])]
    print(f"\nreproduction: {len(rep)} member surfaces, max |diff| {rep['max_abs_diff'].max():.1e}, "
          f"failures {len(bad)}")
    if len(bad):
        print(bad.to_string(index=False))
    return len(bad) == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", nargs="*", default=None)
    ap.add_argument("--levels", nargs="*", type=int, default=[0, 3, 10, 20])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-repro", action="store_true")
    args = ap.parse_args()
    warnings.simplefilter("ignore")

    rpb = load_runner()
    from sdm_robustness.metrics.core import compute_performance_metrics
    from sdm_robustness.pipeline.core import assign_basin_folds, clean_predictors, get_track_columns
    from trustworthy_sdm.analysis import ENTITY_NAME_TO_DIR
    from trustworthy_sdm.regen import assemble_inputs

    with open(REPO / "data/protocol_b_predictor_sets.json") as f:
        pred_sets = json.load(f)
    WORK.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    for entity in (args.entities or rpb.ENTITIES):
        edir = ENTITY_NAME_TO_DIR[entity]
        inputs = None
        for track in rpb.TRACKS:
            for level in args.levels:
                kind = "benchmark" if level == 0 else rpb.AXIS
                cell = f"{edir}__consensus__{track}__{kind}__L{level}"
                cdir = WORK / cell
                if (cdir / "metrics.csv").exists() and not args.force:
                    continue
                if inputs is None:
                    inputs = assemble_inputs(entity=entity, track="combined", axis=rpb.AXIS, master_csv=MASTER)
                    pool = inputs.contamination_pool[["subc_id", "basin_id"]].copy()
                    acc_ids = set(inputs.accessible_area["subc_id"].astype(str))
                    pool["on_surface"] = pool["subc_id"].astype(str).isin(acc_ids)
                    pool.to_parquet(WORK / f"{edir}__pool_sites.parquet", index=False)
                full_cols = clean_predictors(inputs.benchmark, get_track_columns(inputs.benchmark, track),
                                             missing_threshold_pct=rpb.MISSING_THRESHOLD_PCT)
                reduced_cols = pred_sets[f"{entity}|||{track}"]["predictors"]
                cdir.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                pres = training_presences(rpb, inputs, level)
                folds, oof = cv_cell(rpb, inputs, pres, reduced_cols, full_cols,
                                     assign_basin_folds, compute_performance_metrics)
                oof.to_parquet(cdir / "oof.parquet", index=False)
                full_fit_sites(rpb, inputs, pres).to_parquet(cdir / "training_sites.parquet", index=False)
                ident = {"entity": entity, "entity_dir": edir, "track": track, "level": level}
                msg = ""
                if not args.no_repro:
                    rep = repro_check(rpb, inputs, track, level, reduced_cols, full_cols,
                                      REPO / "data/replicate_surfaces_protocol_b" / cell)
                    rep.assign(**ident).to_csv(cdir / "repro.csv", index=False)
                    msg = f", repro max|d| {rep['max_abs_diff'].max():.1e}"
                metric_cols = [c for c in folds.columns if c not in ("fold", "model")]
                g = folds.groupby("model")
                agg = g[metric_cols].mean()
                agg["n_folds"] = g.size()
                agg = agg.reset_index().assign(**ident, n_presence=len(pres),
                                               n_contaminant=int((pres["role"] == "contaminant").sum()))
                agg.to_csv(cdir / "metrics.csv", index=False)
                auc = agg.set_index("model").loc["consensus_mean", "auc"]
                print(f"{cell}: {time.time() - t0:.1f}s, consensus AUC {auc:.3f}{msg}", flush=True)
    print(f"total {time.time() - t_all:.0f}s")
    sys.exit(0 if summarize() else 1)


if __name__ == "__main__":
    main()
