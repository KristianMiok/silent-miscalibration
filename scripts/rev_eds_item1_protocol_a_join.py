"""EDS-2026-0083 revision, item 1 (Protocol A), part 2: performance vs miscalibration.

Joins the out-of-sample (basin-blocked 5-fold CV) performance of every
Protocol A cell with its interval coverage and point-prediction divergence,
to test Reviewer 3's alternative hypothesis that miscalibration is confined
to weak models. Adds the clean-model noise floor for AUC/TSS from the
companion paper's benchmark-stability runs.

Competence gate (out-of-sample; proposed replacement for the in-sample
adequacy check): clean CV AUC >= 0.70.
Miscoverage threshold: 0.90, i.e. nominal 0.95 with the allowance Reviewer 1
suggests for benchmark uncertainty (0.95^2).
A cell is a silent failure if the model is competent, still passes the gate
after contamination, and its uncorrected coverage is below the threshold.

Output: reports/rev_eds/item1_protocol_a_join.csv, one row per
contaminated cell.
"""
import glob
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from trustworthy_sdm.analysis import ENTITY_NAME_TO_DIR  # noqa: E402

OOS = REPO / "reports/rev_eds/item1_protocol_a_oos.csv"
CONF = REPO / "figures/panel_conformal.csv"
SUMM = REPO / "figures/panel_summary.csv"
T5C = REPO / "data/results/task5c_benchmark_stability_array"
OUT = REPO / "reports/rev_eds/item1_protocol_a_join.csv"

AUC_GATE = 0.70
COV_THRESHOLD = 0.90
CELL = ["entity_dir", "algorithm", "track"]
KEYS = CELL + ["level"]
DIRS = set(ENTITY_NAME_TO_DIR.values())
ALGO = {"rf": "random_forest", "random_forest": "random_forest",
        "xgb": "xgboost", "xgboost": "xgboost"}
TRACK = {"local": "local_only", "local_only": "local_only",
         "upstream": "upstream_only", "upstream_only": "upstream_only",
         "combined": "combined"}


def norm(df):
    df = df.copy()
    if "entity_dir" not in df.columns:
        df["entity_dir"] = df["entity"].map(lambda e: e if e in DIRS else ENTITY_NAME_TO_DIR.get(e))
    df["algorithm"] = df["algorithm"].astype(str).str.lower().map(ALGO).fillna(df["algorithm"])
    df["track"] = df["track"].astype(str).map(TRACK).fillna(df["track"])
    if "level" in df.columns:
        df["level"] = pd.to_numeric(df["level"].astype(str).str.lstrip("L"), errors="coerce")
    return df


def load_t5c():
    files = sorted(glob.glob(str(T5C / "*" / "benchmark_stability.parquet")))
    t = norm(pd.concat([pd.read_parquet(f) for f in files], ignore_index=True))
    print("task5c metrics:", sorted(t["metric_name"].unique()))
    t = t[t["algorithm"].isin(["random_forest", "xgboost"])]
    w = t.pivot_table(index=CELL, columns="metric_name", values=["benchmark_mean", "benchmark_sd"])
    w.columns = [f"t5c_{m}_{s.replace('benchmark_', '')}" for s, m in w.columns]
    return w.reset_index()


def main():
    oos = norm(pd.read_csv(OOS))
    clean = (oos[oos["axis"] == "benchmark"][CELL + ["auc_mean", "tss_mean"]]
             .rename(columns={"auc_mean": "auc_clean_oos", "tss_mean": "tss_clean_oos"}))
    cont = (oos[oos["axis"] == "lowacc"][KEYS + ["auc_mean", "tss_mean", "brier_mean", "delta_auc",
                                                 "delta_tss", "delta_brier", "benchmark_presence_n"]]
            .rename(columns={"auc_mean": "auc_cont_oos", "tss_mean": "tss_cont_oos",
                             "brier_mean": "brier_cont_oos"}))
    conf = norm(pd.read_csv(CONF))
    summ = norm(pd.read_csv(SUMM))[KEYS + ["coverage", "median_signed_diff", "mean_abs_diff"]]

    j = (conf.merge(summ, on=KEYS, how="left")
             .merge(cont, on=KEYS, how="left")
             .merge(clean, on=CELL, how="left")
             .merge(load_t5c(), on=CELL, how="left"))
    miss = int(j["auc_clean_oos"].isna().sum())
    print(f"cells: {len(j)}; without OOS match: {miss}")
    if miss:
        print(j.loc[j["auc_clean_oos"].isna(), KEYS].head(10).to_string())
    print(f"max |coverage_uncorrected - coverage|: {(j['coverage_uncorrected'] - j['coverage']).abs().max():.4f}")

    if "t5c_auc_sd" in j.columns:
        j["z_delta_auc"] = j["delta_auc"] / j["t5c_auc_sd"]
    j["competent"] = j["auc_clean_oos"] >= AUC_GATE
    j["passes_gate_contaminated"] = j["auc_cont_oos"] >= AUC_GATE
    j["miscovered"] = j["coverage_uncorrected"] < COV_THRESHOLD
    j["silent_failure"] = j["competent"] & j["passes_gate_contaminated"] & j["miscovered"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    j.round(4).to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(REPO)}\n")

    print("basins per entity (LOBO):")
    print(j.groupby("entity_dir")["n_basins"].median().to_string(), "\n")

    fails = j[~j["competent"]][CELL + ["auc_clean_oos"]].drop_duplicates()
    print(f"configurations failing the OOS competence gate (clean AUC < {AUC_GATE}):")
    print(fails.round(3).to_string(index=False), "\n")

    for x, y in [("auc_clean_oos", "coverage_uncorrected"), ("auc_cont_oos", "coverage_uncorrected"),
                 ("auc_clean_oos", "median_signed_diff"), ("delta_auc", "coverage_uncorrected")]:
        d = j[[x, y]].dropna()
        r, p = spearmanr(d[x], d[y])
        print(f"spearman {x} vs {y}: rho = {r:+.2f}, p = {p:.2g}, n = {len(d)}")
    print()

    g = j.groupby(["competent", "level"])
    tab = pd.DataFrame({
        "n": g.size(),
        "cov_mean": g["coverage_uncorrected"].mean(),
        "cov_min": g["coverage_uncorrected"].min(),
        "cov_conformal": g["coverage_conformal"].mean(),
        "n_miscovered": g["miscovered"].sum(),
        "n_silent": g["silent_failure"].sum(),
    })
    print(tab.round(3).to_string())
    if "z_delta_auc" in j.columns:
        print("\nmedian z of delta AUC relative to clean-replicate sd, by level:")
        print(j.groupby("level")["z_delta_auc"].median().round(2).to_string())


if __name__ == "__main__":
    main()
