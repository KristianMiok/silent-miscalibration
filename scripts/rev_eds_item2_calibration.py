"""EDS-2026-0083 revision, item 2: member calibration for Protocol B (Reviewer 1, points 1 and 4).

Reviewer 1: RF and XGBoost return pseudo-probabilities that should not be
averaged with GLM/GAM probabilities without calibration (Dormann 2020).
All four members are fitted to the same 1:1 presence-background design, so a
common target exists: P(presence record | x) in that design. Each member of
each cell is calibrated on its own out-of-fold predictions from the
basin-blocked 5-fold CV (rev_eds_protocol_b_cv.py), which a practitioner can
do with their own data, and the fitted map is applied to the member's
full-data surface (as in sklearn's CalibratedClassifierCV with ensemble=False).
Primary map: Platt scaling on the logit of the prediction; sensitivity:
isotonic regression. GLM and GAM are calibrated too: penalisation and class
weighting mean they are not calibrated by construction.

Reports, per cell and member, out-of-fold reliability before and after
calibration (Brier score, expected calibration error over 10 equal-width
bins, calibration intercept and slope on the logit scale; the after values
are cross-fitted by fold), then recomputes coverage of the four-member
interval against the calibrated clean benchmark and the submitted
directional statistic on the calibrated surfaces. The uncalibrated
recomputation must reproduce the submitted coverage (exit code 1 otherwise).

Outputs: reports/rev_eds/item2_member_reliability.csv
         reports/rev_eds/item2_calibrated_panel.csv
Calibrated surfaces (record-level) go to data/rev_eds/protocol_b_calibrated_<method>/.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parent.parent
SURF = REPO / "data/replicate_surfaces_protocol_b"
WORK = REPO / "data/rev_eds/protocol_b_cv"
CAL = {m: REPO / f"data/rev_eds/protocol_b_calibrated_{m}" for m in ("platt", "isotonic")}
SUMM = REPO / "figures/panel_summary_protocol_b.csv"
OUT_REL = REPO / "reports/rev_eds/item2_member_reliability.csv"
OUT_PANEL = REPO / "reports/rev_eds/item2_calibrated_panel.csv"
ALGOS = ["glm", "gam", "random_forest", "xgboost"]
REPS = ["rep_00", "rep_01", "rep_02", "rep_03"]
EPS = 1e-6
N_BINS = 5
SCORED = ("local_only", "combined")


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def make_map(p, y, method):
    if method == "platt":
        lr = LogisticRegression(penalty=None, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
        a, b = float(lr.intercept_[0]), float(lr.coef_[0, 0])
        return lambda q: 1.0 / (1.0 + np.exp(-(a + b * logit(q))))
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)
    return lambda q: iso.predict(np.asarray(q, dtype=float))


def reliability(p, y):
    lr = LogisticRegression(penalty=None, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
    b = np.clip((p * 10).astype(int), 0, 9)
    ece = sum(abs(y[b == k].mean() - p[b == k].mean()) * np.mean(b == k) for k in range(10) if np.any(b == k))
    return {"brier": float(np.mean((p - y) ** 2)), "ece": float(ece),
            "cal_intercept": float(lr.intercept_[0]), "cal_slope": float(lr.coef_[0, 0])}


def crossfit(p, y, folds, method):
    out = np.empty_like(p, dtype=float)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        out[te] = make_map(p[tr], y[tr], method)(p[te])
    return out


def load_members(cell_dir):
    frames = [pd.read_parquet(cell_dir / f"{r}.parquet").set_index("subc_id")["predicted_probability"]
              for r in REPS]
    return pd.concat(frames, axis=1)


def panel_stats(C, bench, never):
    lo = np.percentile(C, 2.5, axis=1)
    hi = np.percentile(C, 97.5, axis=1)
    inside = (bench >= lo) & (bench <= hi)
    m = C.mean(axis=1)
    bins = pd.qcut(pd.Series(bench), q=N_BINS, duplicates="drop")
    over = pd.Series(m > bench).groupby(bins, observed=True).mean()
    return {"coverage": float(inside.mean()), "coverage_never_trained": float(inside[never].mean()),
            "median_width": float(np.median(hi - lo)), "mean_diff": float(np.mean(m - bench)),
            "gradient": float(over.iloc[0] - over.iloc[-1])}


def main():
    rel = []
    cells = sorted(d for d in WORK.iterdir() if d.is_dir() and (d / "oof.parquet").exists())
    for cdir in cells:
        oof = pd.read_parquet(cdir / "oof.parquet")
        y = oof["y"].to_numpy(dtype=int)
        folds = oof["fold"].to_numpy()
        for algo, rep in zip(ALGOS, REPS):
            p = oof[algo].to_numpy(dtype=float)
            row = {"cell": cdir.name, "member": algo,
                   **{f"before_{k}": v for k, v in reliability(p, y).items()}}
            surf = pd.read_parquet(SURF / cdir.name / f"{rep}.parquet")
            for method, root in CAL.items():
                row.update({f"{method}_{k}": v for k, v in reliability(crossfit(p, y, folds, method), y).items()})
                out = surf.copy()
                out["predicted_probability"] = make_map(p, y, method)(out["predicted_probability"].to_numpy(dtype=float))
                (root / cdir.name).mkdir(parents=True, exist_ok=True)
                out.to_parquet(root / cdir.name / f"{rep}.parquet", index=False)
            rel.append(row)
    rel = pd.DataFrame(rel)
    parts = rel["cell"].str.split("__", expand=True)
    rel["entity_dir"], rel["track"] = parts[0], parts[2]
    rel["level"] = parts[4].str.lstrip("L").astype(int)
    OUT_REL.parent.mkdir(parents=True, exist_ok=True)
    rel.round(5).to_csv(OUT_REL, index=False)
    cols = ["before_ece", "platt_ece", "isotonic_ece", "before_cal_slope", "platt_cal_slope",
            "before_cal_intercept", "before_brier", "platt_brier"]
    print(f"member reliability, out-of-fold, median over {rel['cell'].nunique()} cells:")
    print(rel.groupby("member")[cols].median().round(3).to_string())

    roots = {"uncalibrated": SURF, **CAL}
    prow = []
    for cdir in sorted(SURF.glob("*__lowacc__L*")):
        edir, _, track, _, lvl = cdir.name.split("__")
        sites = pd.read_parquet(WORK / cdir.name / "training_sites.parquet")
        trained = set(sites.loc[sites["role"].isin(["contaminant", "background"]), "subc_id"].astype(str))
        for method, root in roots.items():
            C = load_members(root / cdir.name)
            B = load_members(root / f"{edir}__consensus__{track}__benchmark__L0")
            idx = C.index.intersection(B.index)
            never = ~pd.Index(idx.astype(str)).isin(list(trained))
            st = panel_stats(C.loc[idx].to_numpy(), B.loc[idx].mean(axis=1).to_numpy(), never)
            prow.append({"entity_dir": edir, "track": track, "level": int(lvl[1:]), "method": method, **st})
    pan = pd.DataFrame(prow)
    pan.round(5).to_csv(OUT_PANEL, index=False)

    summ = pd.read_csv(SUMM)
    summ["level"] = pd.to_numeric(summ["level"].astype(str).str.lstrip("L"))
    u = pan[pan["method"] == "uncalibrated"].merge(summ[["entity_dir", "track", "level", "coverage"]],
                                                   on=["entity_dir", "track", "level"], suffixes=("", "_sub"))
    err = float((u["coverage"] - u["coverage_sub"]).abs().max()) if len(u) else float("nan")
    print(f"\nuncalibrated recomputation vs submitted coverage: {len(u)} cells, max |diff| {err:.2e}")

    order = ["uncalibrated", "platt", "isotonic"]
    for val, label in (("coverage", "coverage"), ("coverage_never_trained", "coverage, never-trained sites"),
                       ("median_width", "median interval width")):
        print(f"\n{label}, mean by track x level:")
        print(pan.pivot_table(index=["track", "level"], columns="method", values=val,
                              aggfunc="mean")[order].round(3).to_string())
    mis = pan.assign(mis=pan["coverage"] < 0.90).pivot_table(index="level", columns="method",
                                                              values="mis", aggfunc="sum")[order]
    print("\ncells with coverage < 0.90 (of 24 per level):")
    print(mis.to_string())
    sc = pan[pan["track"].isin(SCORED)]
    for val in ("gradient", "mean_diff"):
        print(f"\n{val} (submitted directional definition), local + combined, mean by level:")
        print(sc.pivot_table(index="level", columns="method", values=val, aggfunc="mean")[order].round(3).to_string())
    sys.exit(0 if (len(u) > 0 and err < 1e-9) else 1)


if __name__ == "__main__":
    main()
