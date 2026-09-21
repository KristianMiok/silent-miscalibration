"""EDS-2026-0083 revision, item 1 (Protocol B), part 2: performance vs miscalibration.

Same logic as rev_eds_item1_protocol_a_join.py, for the consensus ensemble:
joins the out-of-sample performance of the consensus (mean of the four
members, basin-blocked 5-fold CV from rev_eds_protocol_b_cv.py) with the
submitted per-cell coverage and point-prediction divergence of Protocol B.

Competence gate: clean consensus CV AUC >= 0.70. Miscoverage threshold 0.90.
Silent failure: competent, still >= 0.70 after contamination, coverage < 0.90.
Protocol B has a single contamination draw per cell, so its CV metrics carry
no replicate averaging (unlike Protocol A) and per-cell deltas are noisier.

Output: reports/rev_eds/item1_protocol_b_join.csv, one row per contaminated cell.
"""
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from trustworthy_sdm.analysis import ENTITY_NAME_TO_DIR  # noqa: E402

OOS = REPO / "reports/rev_eds/item1_protocol_b_oos.csv"
CONF = REPO / "figures/panel_conformal_protocol_b.csv"
SUMM = REPO / "figures/panel_summary_protocol_b.csv"
OUT = REPO / "reports/rev_eds/item1_protocol_b_join.csv"
AUC_GATE = 0.70
COV_THRESHOLD = 0.90
MODEL = "consensus_mean"
CELL = ["entity_dir", "track"]
KEYS = CELL + ["level"]
DIRS = set(ENTITY_NAME_TO_DIR.values())


def norm(df):
    df = df.copy()
    if "entity_dir" not in df.columns:
        df["entity_dir"] = df["entity"].map(lambda e: e if e in DIRS else ENTITY_NAME_TO_DIR.get(e))
    df["track"] = df["track"].astype(str).replace({"local": "local_only", "upstream": "upstream_only"})
    df["level"] = pd.to_numeric(df["level"].astype(str).str.lstrip("L"), errors="coerce")
    return df


def main():
    oos = norm(pd.read_csv(OOS))
    oos = oos[oos["model"] == MODEL]
    clean = (oos[oos["level"] == 0][CELL + ["auc", "tss"]]
             .rename(columns={"auc": "auc_clean_oos", "tss": "tss_clean_oos"}))
    cont = (oos[oos["level"] > 0][KEYS + ["auc", "tss", "brier", "n_presence", "n_contaminant"]]
            .rename(columns={"auc": "auc_cont_oos", "tss": "tss_cont_oos", "brier": "brier_cont_oos"}))
    conf = norm(pd.read_csv(CONF))
    summ = norm(pd.read_csv(SUMM))[KEYS + ["coverage", "median_signed_diff", "mean_abs_diff"]]
    j = (conf.merge(summ, on=KEYS, how="left").merge(cont, on=KEYS, how="left")
             .merge(clean, on=CELL, how="left"))
    miss = int(j["auc_clean_oos"].isna().sum() + j["auc_cont_oos"].isna().sum())
    print(f"cells: {len(j)}; missing OOS matches: {miss}")
    print(f"max |coverage_uncorrected - coverage|: {(j['coverage_uncorrected'] - j['coverage']).abs().max():.4f}")

    j["delta_auc"] = j["auc_cont_oos"] - j["auc_clean_oos"]
    j["competent"] = j["auc_clean_oos"] >= AUC_GATE
    j["passes_gate_contaminated"] = j["auc_cont_oos"] >= AUC_GATE
    j["miscovered"] = j["coverage_uncorrected"] < COV_THRESHOLD
    j["silent_failure"] = j["competent"] & j["passes_gate_contaminated"] & j["miscovered"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    j.round(4).to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(REPO)}\n")

    fails = j[~j["competent"]][CELL + ["auc_clean_oos"]].drop_duplicates()
    print(f"entity x track failing the OOS competence gate (clean consensus AUC < {AUC_GATE}):")
    print(fails.round(3).to_string(index=False) if len(fails) else "none", "\n")

    for lvl, d in j.groupby("level"):
        for x in ["auc_clean_oos", "auc_cont_oos", "delta_auc"]:
            dd = d[[x, "coverage_uncorrected"]].dropna()
            r, p = spearmanr(dd[x], dd["coverage_uncorrected"])
            print(f"L{lvl:.0f} {x} vs coverage: rho = {r:+.2f}, p = {p:.2g}, n = {len(dd)}")
    print()

    g = j.groupby(["competent", "level"])
    tab = pd.DataFrame({"n": g.size(), "cov_mean": g["coverage_uncorrected"].mean(),
                        "cov_min": g["coverage_uncorrected"].min(),
                        "cov_conformal": g["coverage_conformal"].mean(),
                        "n_miscovered": g["miscovered"].sum(), "n_silent": g["silent_failure"].sum()})
    print(tab.round(3).to_string(), "\n")
    c = j[j["competent"]]
    print(c.pivot_table(index="track", columns="level", values="coverage_uncorrected",
                        aggfunc="mean").round(3).to_string())


if __name__ == "__main__":
    main()
