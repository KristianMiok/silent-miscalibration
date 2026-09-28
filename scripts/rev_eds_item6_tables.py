"""EDS-2026-0083 revision: item 6 checks, calibration audit and supplement tables.

Sections (aggregates only, written to reports/rev_eds/):
  1. calibration audit: intercept and slope of every Platt member map (fitted on
     the clean out-of-fold predictions) with the member's clean out-of-fold AUC;
     slopes <= 0.05 flag members without out-of-fold skill, which Platt maps to a
     near-constant or inverted surface -> calibration_audit.csv
  2. conformal per sub-catchment (Protocol B, uncalibrated and Platt): the fold
     loop of evaluate_protocol_b.conformal_row rebuilt with the repository's
     nonconformity_scores and conformal_quantile, checked against
     protocol_b_platt_panel.csv; per condition the Spearman correlation between
     per-sub-catchment width inflation and divergence (R3.14)
     -> r314_inflation_divergence.csv
  3. conditions left unchanged by the correction (R3.13), by level, track and
     unit, with unit descriptors -> r313_unchanged.csv
  4. where the shift peaks versus where the contaminants sit: per unit and track
     at L20 (Platt), the bin of largest excess divergence and the bin holding the
     median clean calibrated suitability of low-accuracy sub-catchments
     -> peak_vs_contaminants.csv
  5. Table 1 (clean out-of-fold AUC/TSS: Protocol B calibrated consensus,
     Protocol A RF and XGBoost) and Table S1 (per-unit counts and composition)
     -> table1.csv, table_s1_units.csv
"""
import importlib.util
import os
import sys
import time
import warnings
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
PLATT = REPO / "data/rev_eds/protocol_b_platt"
WORK = REPO / "data/rev_eds/protocol_b_cv"
OUT = REPO / "reports/rev_eds"
UPSTREAM_V1 = REPO.parent / "sdm-robustness-v1.0"
ALGOS = ["glm", "gam", "random_forest", "xgboost"]
REPS = ["rep_00", "rep_01", "rep_02", "rep_03"]
EPS = 1e-6
SHORT = {"Procambarus_clarkii_alien": "Pcla_a", "Pacifastacus_leniusculus_alien": "Plen",
         "Faxonius_limosus_alien": "Flim", "Astacus_astacus": "Aast",
         "Procambarus_clarkii_native": "Pcla_n", "Pontastacus_leptodactylus_pooled": "Plep",
         "Austropotamobius_torrentium_pooled": "Ator", "Austropotamobius_fulcisianus_pooled": "Aful"}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def members(root, cell):
    return pd.concat([pd.read_parquet(root / cell / f"{r}.parquet").set_index("subc_id")["predicted_probability"]
                      for r in REPS], axis=1)


def main():
    warnings.simplefilter("ignore")
    evp = load_module("evaluate_protocol_b", REPO / "scripts/evaluate_protocol_b.py")
    rpb = load_module("run_protocol_b", REPO / "scripts/run_protocol_b.py")
    from sdm_robustness.pipeline.core import clean_predictors, get_track_columns
    from trustworthy_sdm.analysis import ENTITY_NAME_TO_DIR as E2D
    from trustworthy_sdm.conformal import _basin_id_lookup, conformal_quantile, nonconformity_scores
    from trustworthy_sdm.regen import assemble_inputs
    entities, tracks, levels = rpb.ENTITIES, rpb.TRACKS, rpb.LEVELS
    alpha = evp.ALPHA
    inv = {v: k for k, v in SHORT.items()}
    t0 = time.time()

    om = pd.read_csv(OUT / "item1_protocol_b_oos.csv")
    audit = []
    for e in entities:
        for t in tracks:
            oof = pd.read_parquet(WORK / f"{E2D[e]}__consensus__{t}__benchmark__L0" / "oof.parquet")
            y = oof["y"].to_numpy(dtype=int)
            for a in ALGOS:
                lr = LogisticRegression(penalty=None, max_iter=1000).fit(
                    logit(oof[a].to_numpy(dtype=float)).reshape(-1, 1), y)
                sel = om[(om["entity"] == e) & (om["track"] == t) & (om["level"] == 0) & (om["model"] == a)]
                audit.append({"entity_dir": E2D[e], "track": t, "member": a,
                              "intercept": float(lr.intercept_[0]), "slope": float(lr.coef_[0, 0]),
                              "member_oof_auc": float(sel["auc"].iloc[0]) if len(sel) else np.nan})
    audit = pd.DataFrame(audit)
    audit.round(4).to_csv(OUT / "calibration_audit.csv", index=False)
    flag = audit[audit["slope"] <= 0.05]
    print(f"1. calibration audit: {len(audit)} member maps; slope {audit['slope'].min():.2f} to "
          f"{audit['slope'].max():.2f}; {len(flag)} with slope <= 0.05")
    if len(flag):
        print(flag.round(3).to_string(index=False))
    print("   median slope by member and track:")
    print(audit.pivot_table(index="member", columns="track", values="slope", aggfunc="median").round(2).to_string())

    panel = pd.read_csv(OUT / "protocol_b_platt_panel.csv")
    r314, check = [], 0.0
    for method, root in (("uncalibrated", SURF), ("platt", PLATT)):
        for e in entities:
            basins = _basin_id_lookup(e)
            edir = E2D[e]
            for t in tracks:
                B = members(root, f"{edir}__consensus__{t}__benchmark__L0").mean(axis=1)
                for lvl in levels:
                    ens = members(root, f"{edir}__consensus__{t}__lowacc__L{lvl}")
                    shared = ens.index.intersection(B.index).intersection(basins.index)
                    ens_a, bench_a, basin_a = ens.loc[shared], B.loc[shared], basins.loc[shared]
                    lo = ens_a.quantile(alpha / 2, axis=1)
                    hi = ens_a.quantile(1 - alpha / 2, axis=1)
                    qhat = pd.Series(np.nan, index=shared)
                    for tb in basin_a.dropna().unique():
                        is_test = basin_a == tb
                        cal = nonconformity_scores(bench_a[~is_test], lo[~is_test], hi[~is_test])
                        if len(cal):
                            qhat[is_test] = conformal_quantile(cal, alpha=alpha)
                    ok = qhat.notna()
                    covc = float(((bench_a >= lo - qhat) & (bench_a <= hi + qhat))[ok].mean())
                    ref = panel[(panel["method"] == method) & (panel["entity"] == e) & (panel["track"] == t)
                                & (panel["level"] == lvl)]["coverage_conformal"].iloc[0]
                    check = max(check, abs(covc - ref))
                    width = hi - lo
                    use = ok & (width > 0)
                    infl = ((width + 2 * qhat) / width)[use]
                    div = (ens_a.mean(axis=1) - bench_a)[use]
                    varies = infl.nunique() > 1
                    r314.append({"method": method, "entity_dir": edir, "track": t, "level": lvl,
                                 "rho_signed": spearmanr(infl, div)[0] if varies else np.nan,
                                 "rho_abs": spearmanr(infl, div.abs())[0] if varies else np.nan})
    r314 = pd.DataFrame(r314)
    r314.round(4).to_csv(OUT / "r314_inflation_divergence.csv", index=False)
    print(f"\n2. per-sub-catchment conformal rebuilt ({time.time() - t0:.0f}s); max |coverage diff| vs panel {check:.2e}")
    for method in ("uncalibrated", "platt"):
        d = r314[r314["method"] == method].dropna(subset=["rho_signed"])
        print(f"   {method}: {len(d)} conditions with varying inflation; rho(inflation, divergence) < 0 in "
              f"{int((d['rho_signed'] < 0).sum())}, median {d['rho_signed'].median():+.2f}; "
              f"rho(inflation, |divergence|) < 0 in {int((d['rho_abs'] < 0).sum())}, median {d['rho_abs'].median():+.2f}")

    comp = pd.read_csv(OUT / "item4_composition.csv")
    comp["entity_dir"] = comp["entity"].map(E2D)
    meta = pd.read_csv(OUT / "item4b_metadata_by_class.csv")
    fo = meta[meta["taxon"] != "ALL"].pivot_table(index="taxon", columns="class", values="strahler_1_share", aggfunc="mean")
    fo["first_order_enrichment"] = fo["low"] - fo["high"]
    fo.index = fo.index.map(inv)
    p = panel.copy()
    p["unchanged"] = (p["width_inflation_factor"] - 1).abs() < 1e-9
    rows = []
    for method in ("uncalibrated", "platt"):
        pm = p[p["method"] == method]
        w = pm.loc[pm["coverage_conformal"].idxmin()]
        print(f"\n3. {method}: unchanged in {int(pm['unchanged'].sum())} of 72; by level "
              f"{pm.groupby('level')['unchanged'].sum().to_dict()}; by track {pm.groupby('track')['unchanged'].sum().to_dict()}; "
              f"lowest post-conformal: {w['entity_dir']} {w['track']} L{w['level']} {w['coverage_conformal']:.3f}")
        g = pm.groupby("entity_dir")
        unit = pd.DataFrame({"n_unchanged": g["unchanged"].sum(), "min_coverage": g["coverage"].min(),
                             "mean_coverage": g["coverage"].mean()})
        unit = unit.join(fo["first_order_enrichment"]).join(
            comp.set_index("entity_dir")[["auc_low_vs_high_position", "separability_auc_local", "n_high"]])
        print(unit.round(3).to_string())
        print(f"   across units: rho(first-order enrichment, mean coverage) = "
              f"{spearmanr(unit['first_order_enrichment'], unit['mean_coverage'])[0]:+.2f}; "
              f"rho(position AUC, mean coverage) = {spearmanr(unit['auc_low_vs_high_position'], unit['mean_coverage'])[0]:+.2f}; "
              f"rho(first-order enrichment, n unchanged) = {spearmanr(unit['first_order_enrichment'], unit['n_unchanged'])[0]:+.2f}")
        rows.append(unit.reset_index().assign(method=method))
    pd.concat(rows).round(4).to_csv(OUT / "r313_unchanged.csv", index=False)

    bins = pd.read_csv(OUT / "protocol_b_platt_bins.csv")
    pk = []
    for e in entities:
        edir = E2D[e]
        pool = pd.read_parquet(WORK / f"{edir}__pool_sites.parquet")
        pool_ids = set(pool.loc[pool["on_surface"], "subc_id"].astype(str))
        for t in tracks:
            b0 = f"{edir}__consensus__{t}__benchmark__L0"
            sites = pd.read_parquet(WORK / b0 / "training_sites.parquet")
            bg = set(sites.loc[sites["role"] == "background", "subc_id"].astype(str))
            Bp = members(PLATT, b0).mean(axis=1)
            Bp.index = Bp.index.astype(str)
            med = float(Bp.reindex(sorted(pool_ids - bg)).dropna().median())
            bt = bins[(bins["method"] == "platt") & (bins["entity_dir"] == edir) & (bins["track"] == t)
                      & (bins["level"] == 20)].sort_values("bin").reset_index(drop=True)
            peak = int(bt.loc[bt["excess"].idxmax(), "bin"])
            cbin = int(bt["bin"].iloc[min(int(np.searchsorted(bt["bench_max"].to_numpy(), med)), len(bt) - 1)])
            pk.append({"entity_dir": edir, "track": t, "median_clean_suitability_low_sites": med,
                       "contaminant_bin": cbin, "peak_excess_bin": peak, "n_bins": len(bt)})
    pk = pd.DataFrame(pk)
    pk.round(4).to_csv(OUT / "peak_vs_contaminants.csv", index=False)
    print("\n4. bin of the median low-accuracy site vs bin of the largest excess (Platt, L20):")
    print(pk.pivot_table(index="entity_dir", columns="track", values=["contaminant_bin", "peak_excess_bin"],
                         aggfunc="first").to_string())
    print(f"   exact match {int((pk['contaminant_bin'] == pk['peak_excess_bin']).sum())} of {len(pk)}; "
          f"within one bin {int(((pk['contaminant_bin'] - pk['peak_excess_bin']).abs() <= 1).sum())}; "
          f"spearman {spearmanr(pk['contaminant_bin'], pk['peak_excess_bin'])[0]:+.2f}")

    cfg = pd.read_csv(UPSTREAM_V1 / "config/final_panel.csv").set_index("entity")
    s1 = []
    for e in entities:
        inp = assemble_inputs(entity=e, track="combined", axis="lowacc", master_csv=MASTER)
        bch, pool, acc = inp.benchmark, inp.contamination_pool, inp.accessible_area
        row = {"entity": e, "entity_dir": E2D[e], "treatment": cfg.loc[e, "treatment"],
               "n_presences": len(bch), "n_background": len(bch), "n_low_accuracy_pool": len(pool),
               "domain_subcatchments": len(acc), "domain_basins": int(acc["basin_id"].nunique()),
               "benchmark_basins": int(bch["basin_id"].nunique())}
        for t in tracks:
            cols = get_track_columns(bch, t)
            row[f"predictors_{t}_A"] = len(clean_predictors(bch, cols))
            row[f"predictors_{t}_B"] = len(clean_predictors(bch, cols, missing_threshold_pct=70.0))
        s1.append(row)
    s1 = pd.DataFrame(s1).merge(comp[["entity_dir", "low_share", "low_in_occupied_basins", "separability_auc_local",
                                      "auc_low_vs_high_position", "surface_median_all_sites",
                                      "surface_median_low_sites", "oof_median_high_presences"]], on="entity_dir")
    ms = meta[meta["taxon"] != "ALL"].copy()
    ms["entity_dir"] = ms["taxon"].map(inv)
    wide = ms.pivot_table(index="entity_dir", columns="class", aggfunc="mean",
                          values=["strahler_1_share", "snap_m_median", "snap_gt_200m_share", "year_median", "lake_share"])
    wide.columns = [f"{v}_{c}" for v, c in wide.columns]
    s1 = s1.merge(wide.reset_index(), on="entity_dir")
    s1.round(4).to_csv(OUT / "table_s1_units.csv", index=False)

    oosb = pd.read_csv(OUT / "protocol_b_platt_oos.csv")
    tb = oosb[(oosb["method"] == "platt") & (oosb["level"] == 0)].pivot_table(
        index="entity_dir", columns="track", values=["auc", "tss"], aggfunc="mean")
    tb.columns = [f"B_{m}_{t}" for m, t in tb.columns]
    oa = pd.read_csv(OUT / "item1_protocol_a_oos.csv")
    oa = oa[oa["axis"] == "benchmark"].drop(columns=["axis"])
    oa["entity_dir"] = oa["entity"].map(E2D)
    ta = oa.pivot_table(index="entity_dir", columns=["algorithm", "track"], values=["auc_mean", "tss_mean"], aggfunc="mean")
    ta.columns = [f"A_{alg}_{m.replace('_mean', '')}_{t}" for m, alg, t in ta.columns]
    t1 = tb.join(ta).join(s1.set_index("entity_dir")[["n_presences"]])
    t1["N_total"] = 2 * t1["n_presences"]
    t1.round(3).to_csv(OUT / "table1.csv")
    print(f"\n5. tables written ({time.time() - t0:.0f}s). Table S1 (key columns):")
    print(s1.set_index(s1["entity_dir"].map(SHORT))[
        ["treatment", "n_presences", "n_low_accuracy_pool", "domain_subcatchments", "domain_basins", "benchmark_basins",
         "predictors_local_only_A", "predictors_upstream_only_A", "predictors_upstream_only_B",
         "predictors_combined_A", "predictors_combined_B"]].T.to_string())


if __name__ == "__main__":
    main()
