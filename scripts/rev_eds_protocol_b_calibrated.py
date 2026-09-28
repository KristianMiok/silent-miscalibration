"""EDS-2026-0083 revision: Protocol B on calibrated members (fixed-map Platt).

The revised manuscript builds every Protocol B result on calibrated members.
One Platt map per member per configuration, fitted on the clean benchmark's
basin-blocked out-of-fold predictions (as in item 2b), is applied to every
clean, contaminated and null member surface; the calibrated surfaces are
written in the submitted directory structure and evaluated with the paper's
own functions (scripts/evaluate_protocol_b.py: coverage_row, conformal_row).
Gate: the same functions on the submitted (uncalibrated) surfaces must
reproduce figures/panel_summary_protocol_b.csv and
figures/panel_conformal_protocol_b.csv exactly (exit code 1 otherwise).

Outputs (aggregates only), in reports/rev_eds/:
  protocol_b_platt_panel.csv      per condition and method: coverage, widths, conformal
  protocol_b_platt_bins.csv       per condition x bin: N, benchmark range, divergence
                                  (contaminated, null), excess, fraction over-predicted
  protocol_b_platt_oos.csv        per cell and method: out-of-fold AUC/TSS of the consensus
  protocol_b_platt_strata.csv     per condition: coverage on first-order vs higher-order sites
  protocol_b_platt_agreement.csv  per configuration: mean pairwise member correlation (clean)
Calibrated surfaces (record-level) go to data/rev_eds/protocol_b_platt/ and
data/rev_eds/protocol_b_null_platt/ (local only).
"""
import importlib.util
import os
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
MASTER = Path(os.environ.get("TS_MASTER_CSV") or
              REPO.parent / "sdm-robustness/data/raw/combined_data_true_master.csv")
os.environ.setdefault("TS_MASTER_CSV", str(MASTER))
SURF = REPO / "data/replicate_surfaces_protocol_b"
NULL = REPO / "data/rev_eds/protocol_b_null"
WORK = REPO / "data/rev_eds/protocol_b_cv"
PLATT = REPO / "data/rev_eds/protocol_b_platt"
NULL_PLATT = REPO / "data/rev_eds/protocol_b_null_platt"
OUT = REPO / "reports/rev_eds"
ALGOS = ["glm", "gam", "random_forest", "xgboost"]
REPS = ["rep_00", "rep_01", "rep_02", "rep_03"]
EPS = 1e-6
N_BINS = 5
FULL_UNITS = {"Astacus_astacus", "Pacifastacus_leniusculus_alien",
              "Faxonius_limosus_alien", "Austropotamobius_torrentium_pooled"}
SCORED = ("local_only", "combined")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def platt_map(p, y):
    lr = LogisticRegression(penalty=None, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
    a, b = float(lr.intercept_[0]), float(lr.coef_[0, 0])
    return lambda q: 1.0 / (1.0 + np.exp(-(a + b * logit(q))))


def write_calibrated(src_dir, dst_dir, maps):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for algo, rep in zip(ALGOS, REPS):
        d = pd.read_parquet(src_dir / f"{rep}.parquet")
        d["predicted_probability"] = maps[algo](d["predicted_probability"].to_numpy(dtype=float))
        d.to_parquet(dst_dir / f"{rep}.parquet", index=False)


def members(root, cell):
    return pd.concat([pd.read_parquet(root / cell / f"{r}.parquet").set_index("subc_id")["predicted_probability"]
                      for r in REPS], axis=1)


def bin_table(b, dc, dn):
    bins = pd.qcut(b, q=N_BINS, duplicates="drop")
    df = pd.DataFrame({"b": b.to_numpy(), "dc": dc.to_numpy(), "dn": dn.to_numpy()})
    df["oc"] = (df["dc"] > 0).astype(float)
    df["on"] = (df["dn"] > 0).astype(float)
    g = df.groupby(pd.Series(bins.cat.codes.to_numpy()))
    res = pd.DataFrame({"n": g.size(), "bench_min": g["b"].min(), "bench_max": g["b"].max(),
                        "div_cont": g["dc"].mean(), "div_null": g["dn"].mean(),
                        "frac_over_cont": g["oc"].mean(), "frac_over_null": g["on"].mean()})
    res = res.reset_index(names="bin")
    res["excess"] = res["div_cont"] - res["div_null"]
    return res


def main():
    warnings.simplefilter("ignore")
    evp = load_module("evaluate_protocol_b", REPO / "scripts/evaluate_protocol_b.py")
    rpb = load_module("run_protocol_b", REPO / "scripts/run_protocol_b.py")
    from sdm_robustness.metrics.core import compute_performance_metrics
    from trustworthy_sdm.analysis import ENTITY_NAME_TO_DIR
    from trustworthy_sdm.conformal import _basin_id_lookup
    from trustworthy_sdm.regen import assemble_inputs

    entities, tracks, levels = rpb.ENTITIES, rpb.TRACKS, rpb.LEVELS
    t0 = time.time()
    maps = {}
    for e in entities:
        edir = ENTITY_NAME_TO_DIR[e]
        for t in tracks:
            b0 = f"{edir}__consensus__{t}__benchmark__L0"
            oof = pd.read_parquet(WORK / b0 / "oof.parquet")
            y = oof["y"].to_numpy(dtype=int)
            m = {a: platt_map(oof[a].to_numpy(dtype=float), y) for a in ALGOS}
            maps[(e, t)] = m
            for root in (PLATT, NULL_PLATT):
                write_calibrated(SURF / b0, root / b0, m)
            for lvl in levels:
                c = f"{edir}__consensus__{t}__lowacc__L{lvl}"
                write_calibrated(SURF / c, PLATT / c, m)
                write_calibrated(NULL / c, NULL_PLATT / c, m)
    print(f"calibrated surfaces written ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for e in entities:
        for t in tracks:
            for lvl in levels:
                for method, root in (("uncalibrated", SURF), ("platt", PLATT)):
                    cr = evp.coverage_row(e, t, lvl, root)
                    fr = evp.conformal_row(e, t, lvl, root, _basin_id_lookup)
                    rows.append({"method": method, **cr, **{k: v for k, v in fr.items() if k not in cr}})
    panel = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    panel.round(6).to_csv(OUT / "protocol_b_platt_panel.csv", index=False)
    print(f"panel evaluated ({time.time() - t0:.0f}s)", flush=True)

    k3 = ["entity", "track", "level"]
    u = panel[panel["method"] == "uncalibrated"]
    g1 = u.merge(pd.read_csv(REPO / "figures/panel_summary_protocol_b.csv"), on=k3, suffixes=("", "_sub"))
    g2 = u.merge(pd.read_csv(REPO / "figures/panel_conformal_protocol_b.csv"), on=k3, suffixes=("", "_sub"))
    err = max(float((g1["coverage"] - g1["coverage_sub"]).abs().max()),
              float((g2["coverage_conformal"] - g2["coverage_conformal_sub"]).abs().max()),
              float((g2["median_q_hat"] - g2["median_q_hat_sub"]).abs().max()),
              float((g2["width_inflation_factor"] - g2["width_inflation_factor_sub"]).abs().max()))
    print(f"gate: uncalibrated recomputation vs submitted CSVs ({len(g1)} and {len(g2)} conditions): "
          f"max |diff| {err:.2e}")
    if len(g1) != 72 or len(g2) != 72 or err > 1e-9:
        sys.exit(1)

    order = ["uncalibrated", "platt"]
    for val in ("coverage", "coverage_conformal", "median_width"):
        print(f"\n{val}, mean by track x level:")
        print(panel.pivot_table(index=["track", "level"], columns="method", values=val,
                                aggfunc="mean")[order].round(3).to_string())
    for method in order:
        p = panel[panel["method"] == method]
        wif = p["width_inflation_factor"]
        corrected = p[(wif - 1).abs() > 1e-9]
        worst = p.loc[p["coverage"].idxmin()]
        print(f"\n{method}: coverage < 0.90 by level {p.assign(m=p['coverage'] < 0.90).groupby('level')['m'].sum().to_dict()}; "
              f"worst {worst['entity_dir']} {worst['track']} L{worst['level']} {worst['coverage']:.3f}")
        print(f"  conformal: mean {p['coverage_uncorrected'].mean():.3f} -> {p['coverage_conformal'].mean():.3f}, "
              f"post range {p['coverage_conformal'].min():.3f}-{p['coverage_conformal'].max():.3f}, "
              f"post < 0.90: {int((p['coverage_conformal'] < 0.90).sum())}")
        print(f"  unchanged: width inflation 1.00 in {int(((wif - 1).abs() < 1e-9).sum())}, "
              f"median q_hat 0 in {int((p['median_q_hat'].abs() < 1e-12).sum())}, "
              f"coverage unchanged in {int(((p['coverage_conformal'] - p['coverage_uncorrected']).abs() < 1e-12).sum())}")
        if len(corrected):
            print(f"  width inflation where corrected: {corrected['width_inflation_factor'].min():.2f}-"
                  f"{corrected['width_inflation_factor'].max():.2f}x, median {corrected['width_inflation_factor'].median():.2f}x; "
                  f"median by track {corrected.groupby('track')['width_inflation_factor'].median().round(2).to_dict()}")

    orows = []
    for e in entities:
        edir = ENTITY_NAME_TO_DIR[e]
        for t in tracks:
            m = maps[(e, t)]
            for lvl in (0,) + tuple(levels):
                kind = "benchmark" if lvl == 0 else "lowacc"
                oof = pd.read_parquet(WORK / f"{edir}__consensus__{t}__{kind}__L{lvl}" / "oof.parquet")
                scores = {"uncalibrated": oof[ALGOS].mean(axis=1).to_numpy(),
                          "platt": np.column_stack([m[a](oof[a].to_numpy(dtype=float)) for a in ALGOS]).mean(axis=1)}
                for method, sc in scores.items():
                    per = []
                    for f in sorted(oof["fold"].unique()):
                        mask = (oof["fold"] == f).to_numpy()
                        yy = oof.loc[mask, "y"].to_numpy()
                        if len(np.unique(yy)) == 2:
                            per.append(compute_performance_metrics(yy, sc[mask], threshold=0.5))
                    orows.append({"entity_dir": edir, "track": t, "level": lvl, "method": method,
                                  "auc": float(np.mean([r["auc"] for r in per])),
                                  "tss": float(np.mean([r["tss"] for r in per]))})
    oos = pd.DataFrame(orows)
    oos.round(4).to_csv(OUT / "protocol_b_platt_oos.csv", index=False)
    h = pd.read_csv(OUT / "item1_protocol_b_oos.csv")
    h = h[h["model"] == "consensus_mean"][["entity_dir", "track", "level", "auc"]]
    chk = oos[oos["method"] == "uncalibrated"].merge(h, on=["entity_dir", "track", "level"], suffixes=("", "_h"))
    print(f"\nconsistency: uncalibrated consensus OOS AUC vs harness, max |diff| {(chk['auc'] - chk['auc_h']).abs().max():.2e}")
    print("Table 1, calibrated consensus, clean out-of-fold AUC / TSS:")
    t1 = oos[(oos["method"] == "platt") & (oos["level"] == 0)]
    print(t1.pivot_table(index="entity_dir", columns="track", values=["auc", "tss"], aggfunc="mean").round(3).to_string())

    clean = oos[oos["level"] == 0][["entity_dir", "track", "method", "auc"]].rename(columns={"auc": "auc_clean"})
    cont = oos[oos["level"] > 0][["entity_dir", "track", "level", "method", "auc"]].rename(columns={"auc": "auc_cont"})
    j = (panel[["entity_dir", "track", "level", "method", "coverage"]]
         .merge(cont, on=["entity_dir", "track", "level", "method"]).merge(clean, on=["entity_dir", "track", "method"]))
    j["miscovered"] = j["coverage"] < 0.90
    j["silent"] = j["miscovered"] & (j["auc_clean"] >= 0.70) & (j["auc_cont"] >= 0.70)
    print("\nmiscovered and silent conditions by level:")
    print(j.groupby(["method", "level"])[["miscovered", "silent"]].sum().to_string())
    for method in order:
        d = j[(j["method"] == method) & (j["level"] == 20)]
        r, p = spearmanr(d["auc_cont"], d["coverage"])
        print(f"{method}, L20: spearman contaminated AUC vs coverage rho = {r:+.2f} (p = {p:.2g})")

    brows = []
    for method, croot, nroot in (("uncalibrated", SURF, NULL), ("platt", PLATT, NULL_PLATT)):
        for e in entities:
            edir = ENTITY_NAME_TO_DIR[e]
            for t in tracks:
                B = members(croot, f"{edir}__consensus__{t}__benchmark__L0").mean(axis=1)
                for lvl in levels:
                    c = f"{edir}__consensus__{t}__lowacc__L{lvl}"
                    C = members(croot, c).mean(axis=1)
                    N = members(nroot, c).mean(axis=1)
                    idx = B.index.intersection(C.index).intersection(N.index)
                    b = B.loc[idx]
                    brows.append(bin_table(b, C.loc[idx] - b, N.loc[idx] - b)
                                 .assign(method=method, entity_dir=edir, track=t, level=lvl))
    bins = pd.concat(brows, ignore_index=True)
    bins.round(6).to_csv(OUT / "protocol_b_platt_bins.csv", index=False)
    pb = bins[bins["method"] == "platt"]
    print("\nFigure 2 data: excess mean signed divergence over the null, calibrated, mean over units:")
    print(pb.pivot_table(index=["track", "level"], columns="bin", values="excess", aggfunc="mean").round(4).to_string())
    for label, sub in (("all eight units", pb), ("four full-upstream units", pb[pb["entity_dir"].isin(FULL_UNITS)])):
        s = sub[sub["track"].isin(SCORED)]
        print(f"\nlocal + combined, {label}:")
        print(s.pivot_table(index="level", columns="bin", values="excess", aggfunc="mean").round(4).to_string())
        print("  overall excess by level:", s.groupby("level")["excess"].mean().round(4).to_dict())
    s20 = pb[(pb["level"] == 20) & pb["track"].isin(SCORED)]
    print("\nper unit, L20, local + combined, excess by bin:")
    print(s20.pivot_table(index="entity_dir", columns="bin", values="excess", aggfunc="mean").round(4).to_string())

    srows, arows = [], []
    for e in entities:
        edir = ENTITY_NAME_TO_DIR[e]
        acc = assemble_inputs(entity=e, track="combined", axis="lowacc", master_csv=MASTER).accessible_area
        o1 = acc.set_index(acc["subc_id"].astype(str))["strahler"].eq(1)
        for t in tracks:
            b0 = f"{edir}__consensus__{t}__benchmark__L0"
            for method, root in (("uncalibrated", SURF), ("platt", PLATT)):
                M0 = members(root, b0)
                pear = [M0.iloc[:, i].corr(M0.iloc[:, k]) for i, k in combinations(range(4), 2)]
                spear = [M0.iloc[:, i].corr(M0.iloc[:, k], method="spearman") for i, k in combinations(range(4), 2)]
                arows.append({"entity_dir": edir, "track": t, "method": method,
                              "mean_pearson": float(np.mean(pear)), "mean_spearman": float(np.mean(spear))})
                B = M0.mean(axis=1)
                for lvl in levels:
                    C = members(root, f"{edir}__consensus__{t}__lowacc__L{lvl}")
                    idx = C.index.intersection(B.index)
                    Cv = C.loc[idx].to_numpy()
                    b = B.loc[idx].to_numpy()
                    lo, hi = np.percentile(Cv, [2.5, 97.5], axis=1)
                    inside = (b >= lo) & (b <= hi)
                    fo = o1.reindex(pd.Index(idx).astype(str)).fillna(False).to_numpy(dtype=bool)
                    srows.append({"entity_dir": edir, "track": t, "level": lvl, "method": method,
                                  "cov_first_order": float(inside[fo].mean()),
                                  "cov_higher_order": float(inside[~fo].mean()),
                                  "share_first_order": float(fo.mean())})
    strata = pd.DataFrame(srows)
    strata.round(5).to_csv(OUT / "protocol_b_platt_strata.csv", index=False)
    agree = pd.DataFrame(arows)
    agree.round(4).to_csv(OUT / "protocol_b_platt_agreement.csv", index=False)
    print("\ncoverage by stream order, calibrated, mean over units:")
    print(strata[strata["method"] == "platt"].pivot_table(index=["track", "level"],
          values=["cov_first_order", "cov_higher_order"], aggfunc="mean").round(3).to_string())
    for method in order:
        a = agree[agree["method"] == method]
        print(f"member agreement on the clean surface, {method}: mean pairwise Pearson "
              f"{a['mean_pearson'].min():.2f}-{a['mean_pearson'].max():.2f} (median {a['mean_pearson'].median():.2f}); "
              f"Spearman {a['mean_spearman'].min():.2f}-{a['mean_spearman'].max():.2f} (median {a['mean_spearman'].median():.2f})")
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
