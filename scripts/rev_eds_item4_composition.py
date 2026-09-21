"""EDS-2026-0083 revision, item 4: composition of the low-accuracy records (Reviewer 3, Reviewer 2).

Reviewer 3: if low-accuracy records differ from high-accuracy ones in
occurrence or environment, a shift is expected and visible to anyone who
looks. The World of Crayfish records used here are occurrences only (status
Alien, Introduced, Native or Type locality; no absences), so both accuracy
classes enter as presences against the same target-group background. What
can differ is where the records sit. Per taxon this reports:
  counts        high-accuracy (benchmark) and low-accuracy (pool) records and
                distinct sub-catchments; share of pool records inside basins
                occupied by high-accuracy records, on the evaluation surface,
                and on a sub-catchment that also holds a high-accuracy record;
                share with any missing upstream predictor (network position);
  environment   per-predictor standardised mean difference (pool minus
                benchmark, pooled SD) and two-sample KS statistic, summarised
                per track, plus the three largest shifts; and a multivariate
                separability score: basin-blocked 5-fold CV AUC of a random
                forest telling pool from benchmark records, local track;
  surface       the clean consensus value (combined track) at pool
                sub-catchments not drawn as background, against the clean
                model's out-of-fold prediction at held-out high-accuracy
                presences, as an AUC (below 0.5: low-accuracy records sit at
                lower clean suitability); and, at L20, the contaminated
                consensus's out-of-fold AUC for held-out contaminants versus
                held-out retained presences.
It also summarises the non-predictor columns of the prepared tables by
accuracy class, pooled over taxa, as raw material for the metadata part.

Outputs (aggregates only): reports/rev_eds/item4_composition.csv (per taxon)
                           reports/rev_eds/item4_predictor_shift.csv (per taxon x predictor)
"""
import importlib.util
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
MASTER = Path(os.environ.get("TS_MASTER_CSV") or
              REPO.parent / "sdm-robustness/data/raw/combined_data_true_master.csv")
SURF = REPO / "data/replicate_surfaces_protocol_b"
WORK = REPO / "data/rev_eds/protocol_b_cv"
OUT = REPO / "reports/rev_eds/item4_composition.csv"
OUT_PRED = REPO / "reports/rev_eds/item4_predictor_shift.csv"
ALGOS = ["glm", "gam", "random_forest", "xgboost"]
SHORT = {"Procambarus clarkii (alien)": "Pcla_a", "Pacifastacus leniusculus (alien)": "Plen",
         "Faxonius limosus (alien)": "Flim", "Astacus astacus": "Aast",
         "Procambarus clarkii (native)": "Pcla_n", "Pontastacus leptodactylus (pooled)": "Plep",
         "Austropotamobius torrentium (pooled)": "Ator", "Austropotamobius fulcisianus (pooled)": "Aful"}


def load_runner():
    spec = importlib.util.spec_from_file_location("run_protocol_b", REPO / "scripts/run_protocol_b.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def smd(a, b):
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / s) if s > 0 else np.nan


def separability_auc(rpb, bench, pool, cols, assign_basin_folds, build_model, roc_auc_score):
    df = pd.concat([bench[["basin_id"] + cols].assign(y=0), pool[["basin_id"] + cols].assign(y=1)],
                   ignore_index=True)
    med = bench[cols].median(numeric_only=True)
    fold_map = assign_basin_folds(df["basin_id"], n_splits=5, looo_threshold=15)
    f = df["basin_id"].astype(str).map(fold_map)
    aucs = []
    for k in sorted(f.dropna().unique().tolist()):
        tr, te = (f != k).to_numpy(), (f == k).to_numpy()
        if df.loc[tr, "y"].nunique() < 2 or df.loc[te, "y"].nunique() < 2:
            continue
        m = build_model("random_forest", seed=rpb.SEED, n_jobs=-1)
        m.fit(df.loc[tr, cols].fillna(med), df.loc[tr, "y"])
        aucs.append(roc_auc_score(df.loc[te, "y"], m.predict_proba(df.loc[te, cols].fillna(med))[:, 1]))
    return float(np.mean(aucs)) if aucs else np.nan


def main():
    warnings.simplefilter("ignore")
    rpb = load_runner()
    from sklearn.metrics import roc_auc_score
    from sdm_robustness.pipeline.core import assign_basin_folds, build_model, get_track_columns
    from trustworthy_sdm.analysis import ENTITY_NAME_TO_DIR
    from trustworthy_sdm.regen import assemble_inputs

    comp, shifts, meta_b, meta_p = [], [], [], []
    for entity in rpb.ENTITIES:
        edir = ENTITY_NAME_TO_DIR[entity]
        inp = assemble_inputs(entity=entity, track="combined", axis=rpb.AXIS, master_csv=MASTER)
        bench, pool, acc = inp.benchmark, inp.contamination_pool, inp.accessible_area
        b_sub = set(bench["subc_id"].astype(str))
        p_sub = pool["subc_id"].astype(str)
        b_basins = set(bench["basin_id"].dropna().astype(str))
        up_cols = get_track_columns(bench, "upstream_only")
        row = {"entity": entity, "n_high": len(bench), "n_low": len(pool),
               "low_share": len(pool) / (len(pool) + len(bench)),
               "subc_high": len(b_sub), "subc_low": int(p_sub.nunique()),
               "low_in_occupied_basins": float(pool["basin_id"].astype(str).isin(b_basins).mean()),
               "low_on_surface": float(p_sub.isin(set(acc["subc_id"].astype(str))).mean()),
               "low_on_high_subc": float(p_sub.isin(b_sub).mean()),
               "high_any_upstream_missing": float(bench[up_cols].isna().any(axis=1).mean()),
               "low_any_upstream_missing": float(pool[up_cols].isna().any(axis=1).mean())}
        for track in ("local_only", "upstream_only"):
            vals = {}
            for c in get_track_columns(bench, track):
                d = smd(pool[c], bench[c])
                a, b = pool[c].dropna(), bench[c].dropna()
                ks = float(ks_2samp(a, b).statistic) if len(a) > 1 and len(b) > 1 else np.nan
                shifts.append({"entity": entity, "track": track, "predictor": c, "smd": d, "ks": ks})
                vals[c] = d
            s = pd.Series(vals).dropna()
            row[f"{track}_median_abs_smd"] = float(s.abs().median())
            row[f"{track}_share_abs_smd_gt_0.2"] = float((s.abs() > 0.2).mean())
            row[f"{track}_share_abs_smd_gt_0.5"] = float((s.abs() > 0.5).mean())
            top = s.reindex(s.abs().sort_values(ascending=False).index[:3])
            row[f"{track}_top_shifts"] = "; ".join(f"{k} {v:+.2f}" for k, v in top.items())
        row["separability_auc_local"] = separability_auc(
            rpb, bench, pool, get_track_columns(bench, "local_only"),
            assign_basin_folds, build_model, roc_auc_score)

        cell0 = f"{edir}__consensus__combined__benchmark__L0"
        B = pd.concat([pd.read_parquet(SURF / cell0 / f"rep_0{i}.parquet").set_index("subc_id")["predicted_probability"]
                       for i in range(4)], axis=1).mean(axis=1)
        B.index = B.index.astype(str)
        sites = pd.read_parquet(WORK / cell0 / "training_sites.parquet")
        bg = set(sites.loc[sites["role"] == "background", "subc_id"].astype(str))
        keep = [i for i in p_sub.unique() if i in B.index and i not in bg]
        low_vals = B.loc[keep].to_numpy()
        oof0 = pd.read_parquet(WORK / cell0 / "oof.parquet")
        high_vals = oof0.loc[oof0["role"] == "kept", ALGOS].mean(axis=1).to_numpy()
        row["surface_median_all_sites"] = float(B.median())
        row["surface_median_low_sites"] = float(np.median(low_vals)) if len(low_vals) else np.nan
        row["oof_median_high_presences"] = float(np.median(high_vals))
        row["auc_low_vs_high_position"] = float(roc_auc_score(
            np.r_[np.ones(len(low_vals)), np.zeros(len(high_vals))], np.r_[low_vals, high_vals])) if len(low_vals) else np.nan
        o20 = pd.read_parquet(WORK / f"{edir}__consensus__combined__lowacc__L20" / "oof.parquet")
        pr = o20[o20["y"] == 1]
        row["l20_oof_auc_contaminant_vs_kept"] = float(roc_auc_score(
            (pr["role"] == "contaminant").astype(int), pr[ALGOS].mean(axis=1)))
        comp.append(row)
        meta = [c for c in bench.columns if not (c.startswith("l_") or c.startswith("u_"))]
        meta_b.append(bench[meta].assign(entity=entity))
        meta_p.append(pool[meta].assign(entity=entity))
        print(f"{entity}: done", flush=True)

    comp = pd.DataFrame(comp)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    comp.round(4).to_csv(OUT, index=False)
    pd.DataFrame(shifts).round(4).to_csv(OUT_PRED, index=False)

    cols = ["n_high", "n_low", "low_share", "low_in_occupied_basins", "low_on_surface", "low_on_high_subc",
            "high_any_upstream_missing", "low_any_upstream_missing",
            "local_only_median_abs_smd", "local_only_share_abs_smd_gt_0.2", "local_only_share_abs_smd_gt_0.5",
            "upstream_only_median_abs_smd", "upstream_only_share_abs_smd_gt_0.2",
            "separability_auc_local", "surface_median_all_sites", "surface_median_low_sites",
            "oof_median_high_presences", "auc_low_vs_high_position", "l20_oof_auc_contaminant_vs_kept"]
    t = comp.set_index(comp["entity"].map(SHORT))[cols].T
    print("\ncomposition by taxon:")
    print(t.round(3).to_string())
    print("\nlargest predictor shifts (SMD, low minus high accuracy):")
    for _, r in comp.iterrows():
        print(f"{SHORT[r['entity']]}  local: {r['local_only_top_shifts']}  |  upstream: {r['upstream_only_top_shifts']}")

    mb = pd.concat(meta_b, ignore_index=True)
    mp = pd.concat(meta_p, ignore_index=True)
    print("\nnon-predictor columns, pooled over taxa: high-accuracy vs low-accuracy")
    for c in [c for c in mb.columns if c != "entity"]:
        if pd.api.types.is_numeric_dtype(mb[c]):
            print(f"  {c}: numeric; median {mb[c].median():.4g} vs {mp[c].median():.4g}; "
                  f"missing {mb[c].isna().mean():.2f} vs {mp[c].isna().mean():.2f}")
        else:
            nu = int(mb[c].nunique())
            if nu <= 12:
                hb = mb[c].value_counts(normalize=True).head(4).round(2).to_dict()
                hp = mp[c].value_counts(normalize=True).head(4).round(2).to_dict()
                print(f"  {c}: {nu} levels; high {hb} | low {hp}")
            else:
                print(f"  {c}: {mb[c].dtype}, {nu} distinct values")


if __name__ == "__main__":
    main()
