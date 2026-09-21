"""EDS-2026-0083 revision, item 2b: fixed-map member calibration (Reviewer 1, points 1 and 4).

Per-cell calibration (item 2) fits a separate map for every cell, so at L3,
where contaminated and clean surfaces barely differ, the two maps alone move
coverage from 0.98 to 0.92: map noise, not contamination. To test Reviewer
1's specific concern (that the patterns come from averaging members on
different probability scales), this version fits ONE map per member per
taxon x track on the clean benchmark's out-of-fold predictions and applies
that same map to the clean, contaminated and matched-null surfaces. Every
comparison is then made on a common calibrated scale, with an identical
transformation on both sides.

Maps: Platt scaling on the logit (primary) and isotonic regression
(sensitivity); uncalibrated as reference. Per contaminated cell, for the
contaminated and the matched-perturbation null surfaces against the
calibrated clean benchmark: coverage of the four-member interval, the
submitted directional statistic (fraction over-predicted in qcut bins,
lowest minus highest bin) and the mean signed divergence.

Pre-registered reading (fixed before the run), local_only + combined tracks:
  (1) at L10 and L20 the mean coverage deficit (0.95 minus mean coverage)
      under fixed-map Platt is at least half the uncalibrated deficit;
  (2) under fixed-map Platt the excess gradient over the null is positive at
      L3, L10 and L20;
  (3) under fixed-map Platt the excess mean signed divergence over the null
      is positive and increases from L3 to L10 to L20.
If all three hold, the coverage failure and the directional effect are not
artefacts of averaging uncalibrated members. The uncalibrated recomputation
must reproduce the submitted coverage (exit code 1 otherwise).

Output: reports/rev_eds/item2b_fixed_map_calibration.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parent.parent
SURF = REPO / "data/replicate_surfaces_protocol_b"
NULL = REPO / "data/rev_eds/protocol_b_null"
WORK = REPO / "data/rev_eds/protocol_b_cv"
SUMM = REPO / "figures/panel_summary_protocol_b.csv"
OUT = REPO / "reports/rev_eds/item2b_fixed_map_calibration.csv"
ALGOS = ["glm", "gam", "random_forest", "xgboost"]
REPS = ["rep_00", "rep_01", "rep_02", "rep_03"]
EPS = 1e-6
N_BINS = 5
SCORED = ("local_only", "combined")
METHODS = ("uncalibrated", "platt", "isotonic")


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def make_map(p, y, method):
    if method == "uncalibrated":
        return lambda q: np.asarray(q, dtype=float)
    if method == "platt":
        lr = LogisticRegression(penalty=None, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
        a, b = float(lr.intercept_[0]), float(lr.coef_[0, 0])
        return lambda q: 1.0 / (1.0 + np.exp(-(a + b * logit(q))))
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)
    return lambda q: iso.predict(np.asarray(q, dtype=float))


def load_members(cell_dir):
    frames = [pd.read_parquet(cell_dir / f"{r}.parquet").set_index("subc_id")["predicted_probability"]
              for r in REPS]
    return pd.concat(frames, axis=1)


def apply_maps(M, maps):
    return np.column_stack([maps[a](M[:, i]) for i, a in enumerate(ALGOS)])


def stats(C, bench):
    lo = np.percentile(C, 2.5, axis=1)
    hi = np.percentile(C, 97.5, axis=1)
    m = C.mean(axis=1)
    bins = pd.qcut(pd.Series(bench), q=N_BINS, duplicates="drop")
    over = pd.Series(m > bench).groupby(bins, observed=True).mean()
    return {"coverage": float(np.mean((bench >= lo) & (bench <= hi))),
            "gradient": float(over.iloc[0] - over.iloc[-1]),
            "mean_diff": float(np.mean(m - bench))}


def main():
    rows = []
    for bdir in sorted(SURF.glob("*__benchmark__L0")):
        edir, _, track, _, _ = bdir.name.split("__")
        oof = pd.read_parquet(WORK / bdir.name / "oof.parquet")
        y = oof["y"].to_numpy(dtype=int)
        Bdf = load_members(bdir)
        for method in METHODS:
            maps = {a: make_map(oof[a].to_numpy(dtype=float), y, method) for a in ALGOS}
            for lvl in (3, 10, 20):
                cell = f"{edir}__consensus__{track}__lowacc__L{lvl}"
                for source, root in (("contaminated", SURF), ("null", NULL)):
                    Cdf = load_members(root / cell)
                    idx = Cdf.index.intersection(Bdf.index)
                    bench = apply_maps(Bdf.loc[idx].to_numpy(), maps).mean(axis=1)
                    C = apply_maps(Cdf.loc[idx].to_numpy(), maps)
                    rows.append({"entity_dir": edir, "track": track, "level": lvl, "method": method,
                                 "source": source, **stats(C, bench)})
    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.round(5).to_csv(OUT, index=False)

    summ = pd.read_csv(SUMM)
    summ["level"] = pd.to_numeric(summ["level"].astype(str).str.lstrip("L"))
    u = res[(res["method"] == "uncalibrated") & (res["source"] == "contaminated")].merge(
        summ[["entity_dir", "track", "level", "coverage"]], on=["entity_dir", "track", "level"],
        suffixes=("", "_sub"))
    err = float((u["coverage"] - u["coverage_sub"]).abs().max()) if len(u) else float("nan")
    print(f"uncalibrated recomputation vs submitted coverage: {len(u)} cells, max |diff| {err:.2e}")

    order = list(METHODS)
    cont = res[res["source"] == "contaminated"]
    print("\ncoverage (contaminated vs calibrated clean benchmark), mean by track x level:")
    print(cont.pivot_table(index=["track", "level"], columns="method", values="coverage",
                           aggfunc="mean")[order].round(3).to_string())
    mis = cont.assign(mis=cont["coverage"] < 0.90).pivot_table(index="level", columns="method",
                                                                values="mis", aggfunc="sum")[order]
    print("\ncells with coverage < 0.90 (of 24 per level):")
    print(mis.to_string())

    sc = res[res["track"].isin(SCORED)]
    cov = sc[sc["source"] == "contaminated"].pivot_table(index="level", columns="method",
                                                         values="coverage", aggfunc="mean")[order]
    g = sc.pivot_table(index=["method", "level"], columns="source", values="gradient", aggfunc="mean")
    g["excess"] = g["contaminated"] - g["null"]
    md = sc.pivot_table(index=["method", "level"], columns="source", values="mean_diff", aggfunc="mean")
    md["excess"] = md["contaminated"] - md["null"]
    print("\nlocal + combined: gradient (fraction over-predicted, lowest minus highest bin):")
    print(g.round(3).to_string())
    print("\nlocal + combined: mean signed divergence (consensus minus benchmark):")
    print(md.round(4).to_string())

    deficit = 0.95 - cov
    ratios = {lv: float(deficit.loc[lv, "platt"] / deficit.loc[lv, "uncalibrated"]) for lv in (10, 20)}
    c1 = all(r >= 0.5 for r in ratios.values())
    gp = g.loc["platt"]["excess"]
    c2 = bool((gp > 0).all())
    mp = md.loc["platt"]["excess"].sort_index().to_numpy()
    c3 = bool((mp > 0).all() and np.all(np.diff(mp) > 0))
    print(f"\ncriterion 1 (coverage deficit ratio platt/uncalibrated at L10, L20: "
          f"{ratios[10]:.2f}, {ratios[20]:.2f}; need >= 0.5): {'PASS' if c1 else 'FAIL'}")
    print(f"criterion 2 (excess gradient > 0 at every level under platt): {'PASS' if c2 else 'FAIL'}")
    print(f"criterion 3 (excess mean divergence > 0 and increasing under platt): {'PASS' if c3 else 'FAIL'}")
    sys.exit(0 if (len(u) > 0 and err < 1e-9) else 1)


if __name__ == "__main__":
    main()
