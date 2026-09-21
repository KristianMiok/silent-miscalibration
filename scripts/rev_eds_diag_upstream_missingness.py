"""EDS-2026-0083 revision, diagnostic: is the upstream-track signal carried by missingness?

Protocol A inherits the companion paper's predictor screen (sdm-robustness
v1.0 clean_predictors default: at most 30% missing); Protocol B passes 70%,
with median imputation. For four entities the upstream track goes from one
predictor (A) to 80-85 predictors with 30-70% missing values (B). This asks
whether Protocol B's out-of-sample skill on that track comes from the
predictor values or from the missingness pattern itself.

For every entity, clean benchmark (L0), upstream_only track: basin-blocked
5-fold CV with the same folds and background draws as rev_eds_protocol_b_cv.py,
random forest on (a) the median-imputed values (must reproduce the harness)
and (b) the binary missingness indicators alone. Also reports how often
presences and accessible-area sites have any missing upstream value.

Output: reports/rev_eds/diag_upstream_missingness.csv
"""
import importlib.util
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
MASTER = Path(os.environ.get("TS_MASTER_CSV") or
              REPO.parent / "sdm-robustness/data/raw/combined_data_true_master.csv")
OUT = REPO / "reports/rev_eds/diag_upstream_missingness.csv"
TRACK = "upstream_only"


def load_runner():
    spec = importlib.util.spec_from_file_location("run_protocol_b", REPO / "scripts/run_protocol_b.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cv_auc(rpb, pres, acc, make_x, assign_basin_folds, build_model, roc_auc_score):
    fold_map = assign_basin_folds(pres["basin_id"], n_splits=5, looo_threshold=15)
    pres = pres.copy()
    pres["fold"] = pres["basin_id"].astype(str).map(fold_map)
    n_neg = min(int(round(len(pres) * rpb.PA_RATIO)), len(acc))
    aucs = []
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
        m = build_model("random_forest", seed=rpb.SEED, n_jobs=-1)
        m.fit(pd.concat([make_x(p_tr), make_x(n_tr)], axis=0), y_tr)
        p = m.predict_proba(pd.concat([make_x(p_te), make_x(n_te)], axis=0))[:, 1]
        aucs.append(roc_auc_score(y_te, p))
    return float(np.mean(aucs)), len(aucs)


def main():
    warnings.simplefilter("ignore")
    rpb = load_runner()
    from sklearn.metrics import roc_auc_score
    from sdm_robustness.pipeline.core import (assign_basin_folds, build_model,
                                              clean_predictors, get_track_columns)
    from trustworthy_sdm.regen import assemble_inputs

    harness = pd.read_csv(REPO / "reports/rev_eds/item1_protocol_b_oos.csv")
    rows = []
    for entity in rpb.ENTITIES:
        inputs = assemble_inputs(entity=entity, track="combined", axis=rpb.AXIS, master_csv=MASTER)
        b, acc = inputs.benchmark, inputs.accessible_area
        track_cols = get_track_columns(b, TRACK)
        cols = clean_predictors(b, track_cols, missing_threshold_pct=rpb.MISSING_THRESHOLD_PCT)
        med = b[cols].median(numeric_only=True)
        pres = b.reset_index(drop=True)
        auc_val, nf = cv_auc(rpb, pres, acc, lambda d: d[cols].fillna(med),
                             assign_basin_folds, build_model, roc_auc_score)
        auc_na, _ = cv_auc(rpb, pres, acc, lambda d: d[cols].isna().astype(np.int8),
                           assign_basin_folds, build_model, roc_auc_score)
        h = harness[(harness["entity"] == entity) & (harness["track"] == TRACK)
                    & (harness["level"] == 0) & (harness["model"] == "random_forest")]["auc"]
        rows.append({
            "entity": entity,
            "n_pred_30pct": len(clean_predictors(b, track_cols)),
            "n_pred_70pct": len(cols),
            "pres_frac_any_missing": round(float(b[cols].isna().any(axis=1).mean()), 3),
            "acc_frac_any_missing": round(float(acc[cols].isna().any(axis=1).mean()), 3),
            "rf_auc_harness": round(float(h.iloc[0]), 3) if len(h) else np.nan,
            "rf_auc_values": round(auc_val, 3),
            "rf_auc_missingness_only": round(auc_na, 3),
            "n_folds": nf,
        })
        print(f"{entity}: values {auc_val:.3f}, missingness only {auc_na:.3f}", flush=True)
    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    print()
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
