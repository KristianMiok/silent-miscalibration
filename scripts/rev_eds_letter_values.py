"""EDS-2026-0083 revision: values for the response letter (R3.13, R3.14, stream order, member
agreement) and a check of the upstream-only first-order stratum.

From committed CSVs and the submitted Protocol B surfaces:
  R3.13   the conditions left unchanged by conformal calibration (calibrated members), one row
          each with unit, track, level, presences and clean CV AUC, cross-tabulated by level
          and track; the definition used is the one that reproduces the per-unit counts of
          r313_unchanged.csv;
  R3.14   per-condition rank correlation between width inflation and divergence (signed and
          absolute), where it is defined;
  coverage of first-order and higher-order sub-catchments by track and level;
  member agreement, mean pairwise Pearson and Spearman correlation, uncalibrated and Platt;
  upstream check: on the upstream-only track, first-order coverage is exactly 0 or 1 in every
          condition, as if all first-order sub-catchments shared one prediction. For each unit
          and member file, the share of sub-catchments at the most frequent predicted value of
          the clean upstream-only surface is set against the first-order share (local-only
          surface as contrast).
Outputs: reports/rev_eds/r313_unchanged_conditions.csv,
         reports/rev_eds/upstream_first_order_check.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
R = REPO / "reports/rev_eds"
SURF = REPO / "data/replicate_surfaces_protocol_b"
REPS = ["rep_00", "rep_01", "rep_02", "rep_03"]
TRACKS = ("local_only", "upstream_only", "combined")


def main():
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    pan = pd.read_csv(R / "protocol_b_platt_panel.csv")
    pan = pan[pan["method"] == "platt"].reset_index(drop=True)
    t1 = pd.read_csv(R / "table1.csv")
    auc = t1.melt(id_vars=["entity_dir", "n_presences"], value_vars=[f"B_auc_{t}" for t in TRACKS],
                  var_name="track", value_name="clean_cv_auc")
    auc["track"] = auc["track"].str.replace("B_auc_", "", regex=False)
    r313 = pd.read_csv(R / "r313_unchanged.csv")
    r313 = r313[r313["method"] == "platt"].set_index("entity_dir")["n_unchanged"]
    cands = {
        "median width unchanged": pd.Series(np.isclose(pan["median_width_conformal"], pan["median_width_uncorrected"])),
        "width factor 1.00": pan["width_inflation_factor"].round(2) == 1.0,
        "median q_hat 0": pan["median_q_hat"] == 0,
        "coverage unchanged": pd.Series(np.isclose(pan["coverage_conformal"], pan["coverage_uncorrected"])),
    }
    chosen = None
    for name, m in cands.items():
        per = pan.loc[m.to_numpy()].groupby("entity_dir").size().reindex(r313.index, fill_value=0)
        ok = bool((per == r313).all())
        print(f"R3.13 definition '{name}': {int(m.sum())} conditions; per-unit counts match r313_unchanged.csv: {ok}")
        if ok and chosen is None:
            chosen = name
    if chosen is None:
        chosen = "median width unchanged"
        print("  no candidate reproduces r313_unchanged.csv; using 'median width unchanged'")
    un = (pan.loc[cands[chosen].to_numpy(), ["entity_dir", "track", "level", "coverage_uncorrected"]]
          .merge(auc, on=["entity_dir", "track"]).sort_values(["level", "track", "entity_dir"]))
    un.round(3).to_csv(R / "r313_unchanged_conditions.csv", index=False)
    print(f"\nR3.13 ({chosen}), unchanged conditions by track and level:")
    print(pd.crosstab(un["track"], un["level"], margins=True).to_string())
    print(un.round(3).to_string(index=False))

    r = pd.read_csv(R / "r314_inflation_divergence.csv")
    r = r[r["method"] == "platt"]
    d = r.dropna(subset=["rho_signed"])
    print(f"\nR3.14 (Platt): rank correlation defined in {len(d)} of {len(r)} conditions")
    for col, lab in (("rho_signed", "inflation vs signed divergence"), ("rho_abs", "inflation vs |divergence|")):
        print(f"  {lab}: negative in {int((d[col] < 0).sum())} of {len(d)}, median {d[col].median():+.2f}; by track: "
              + "; ".join(f"{t} {int((g[col] < 0).sum())}/{len(g)} negative, median {g[col].median():+.2f}"
                          for t, g in d.groupby("track")))

    s = pd.read_csv(R / "protocol_b_platt_strata.csv")
    s = s[s["method"] == "platt"]
    print("\nstream-order coverage (Platt), mean over units:")
    print(s.groupby(["track", "level"])[["cov_first_order", "cov_higher_order"]].mean().round(3).to_string())
    u = s[s["track"] == "upstream_only"]
    print("upstream-only first-order coverage per unit:")
    print(u.pivot_table(index="entity_dir", columns="level", values="cov_first_order").round(3).to_string())
    print(f"  values strictly between 0 and 1: {int(((u['cov_first_order'] > 0) & (u['cov_first_order'] < 1)).sum())} of {len(u)}")

    a = pd.read_csv(R / "protocol_b_platt_agreement.csv")
    print("\nmember agreement, mean pairwise correlation of the four members (24 configurations):")
    for m in ("uncalibrated", "platt"):
        g = a[a["method"] == m]
        for col in ("mean_pearson", "mean_spearman"):
            print(f"  {m:12s} {col:14s} {g[col].min():.2f}-{g[col].max():.2f}, median {g[col].median():.2f}")
    w = a.pivot_table(index=["entity_dir", "track"], columns="method", values="mean_spearman")
    print("  configurations where Platt changes the Spearman mean:")
    print(w[(w["platt"] - w["uncalibrated"]).abs() > 1e-6].round(3).to_string())

    share1 = s[s["level"] == 3].groupby("entity_dir")["share_first_order"].first()
    rows = []
    for edir, sh in share1.items():
        for t in ("upstream_only", "local_only"):
            cell = SURF / f"{edir}__consensus__{t}__benchmark__L0"
            for rep in REPS:
                p = pd.read_parquet(cell / f"{rep}.parquet")["predicted_probability"].round(10)
                rows.append({"entity_dir": edir, "track": t, "member_file": rep, "first_order_share": sh,
                             "modal_value_share": float(p.value_counts(normalize=True).iloc[0]),
                             "n_distinct": int(p.nunique())})
    chk = pd.DataFrame(rows)
    chk.round(4).to_csv(R / "upstream_first_order_check.csv", index=False)
    print("\nupstream check: share of sub-catchments at the most frequent clean prediction, vs first-order share")
    print(chk.pivot_table(index=["entity_dir", "first_order_share"], columns=["track", "member_file"],
                          values="modal_value_share").round(3).to_string())


if __name__ == "__main__":
    main()
