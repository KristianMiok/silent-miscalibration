"""EDS-2026-0083 revision, item 3c: matched-perturbation null for the directional analysis (Protocol B).

The directional statistic of Fig. 2 (fraction of sites where the contaminated
consensus mean exceeds the clean benchmark mean, in equal-count benchmark
bins) can show a low-to-high gradient without any contamination: binning by
one estimate and comparing another is exposed to regression to the mean, and
bounded [0, 1] predictions mostly move up near 0 and down near 1. At 3%
contamination the gradient already has about 70% of its 20% size, which is
what such an artefact would look like.

Null: identical construction to each contaminated cell (same retained
high-accuracy presences, same background draw, same model seeds, same
training size), except that the replaced records are duplicates drawn from
the retained high-accuracy records instead of low-accuracy pool records.
Only the source of the replacement differs.

The directional statistic uses the submitted definition exactly (consensus
mean vs benchmark mean, pd.qcut 5 bins, duplicates dropped); the script
checks that it reproduces figures/asymmetry_protocol_b_5bin_L10/L20.csv and
exits with code 1 otherwise.

Pre-registered reading (fixed before the run), on the local_only and combined
tracks (upstream_only is reported but not scored: its plateaued benchmark
surfaces make the bins unstable):
  (1) the contaminated gradient (lowest minus highest bin fraction
      over-predicted, panel mean) exceeds the null gradient at L3, L10, L20;
  (2) the excess increases from L3 to L10 to L20.
Also reported: mean signed divergence by bin (the continuous measure asked by
Reviewer 1) and a Bland-Altman binning by the mean of the two predictions.

Outputs: reports/rev_eds/item3_null_directional_bins.csv
         reports/rev_eds/item3_null_directional_summary.csv
Null surfaces (record-level) go to data/rev_eds/protocol_b_null/ (local).
"""
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
SURF = REPO / "data/replicate_surfaces_protocol_b"
NULL = REPO / "data/rev_eds/protocol_b_null"
OUT_BINS = REPO / "reports/rev_eds/item3_null_directional_bins.csv"
OUT_SUMM = REPO / "reports/rev_eds/item3_null_directional_summary.csv"
N_BINS = 5
SCORED = ("local_only", "combined")


def load_runner():
    spec = importlib.util.spec_from_file_location("run_protocol_b", REPO / "scripts/run_protocol_b.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def null_presences(bench, level, seed):
    """Same retained set as contaminate_presence_set; the replaced records are
    duplicates of retained high-accuracy records."""
    n_replace = int(round(len(bench) * level / 100.0))
    kept = bench.sample(n=len(bench) - n_replace, replace=False, random_state=seed).copy()
    dup = kept.sample(n=n_replace, replace=True, random_state=seed + 1).copy()
    return pd.concat([kept, dup], ignore_index=True)


def fit_consensus(rpb, inputs, pres, reduced_cols, full_cols):
    """consensus_surfaces_for_cell with a given presence set (same background and seeds)."""
    bench, acc = inputs.benchmark, inputs.accessible_area
    out = {}
    for algo in rpb.CONSENSUS_ORDER:
        cols = reduced_cols if algo in {"glm", "gam"} else full_cols
        med = bench[cols].median(numeric_only=True)
        pres_x = pres[cols].fillna(med)
        n_neg = min(int(round(len(pres_x) * rpb.PA_RATIO)), len(acc))
        neg = acc.sample(n=n_neg, replace=False, random_state=rpb.SEED)
        x_train = pd.concat([pres_x, neg[cols].fillna(med)], axis=0)
        y_train = np.array([1] * len(pres_x) + [0] * n_neg)
        pred = rpb.fit_one_algorithm(algo, x_train, y_train, acc[cols].fillna(med), rpb.SEED)
        out[algo] = pd.Series(np.clip(pred, 0, 1), index=acc["subc_id"].values)
    return out


def load_members(cell_dir):
    frames = [pd.read_parquet(p).set_index("subc_id")["predicted_probability"]
              for p in sorted(cell_dir.glob("rep_*.parquet"))]
    return pd.concat(frames, axis=1)


def directional(ens, bench):
    shared = ens.index.intersection(bench.index)
    m = ens.loc[shared].mean(axis=1)
    b = bench.loc[shared]
    d = m - b
    over = (m > b).astype(float)
    out = []
    for name, key in (("benchmark", pd.qcut(b, q=N_BINS, duplicates="drop")),
                      ("bland_altman", pd.qcut(((m + b) / 2).rank(method="first"), q=N_BINS, labels=False))):
        g_over = over.groupby(key, observed=True).mean()
        g_diff = d.groupby(key, observed=True).mean()
        g_n = d.groupby(key, observed=True).size()
        out.append(pd.DataFrame({"binning": name, "bin": np.arange(len(g_over)),
                                 "frac_over": g_over.to_numpy(), "mean_diff": g_diff.to_numpy(),
                                 "n": g_n.to_numpy()}))
    return out


def main():
    warnings.simplefilter("ignore")
    rpb = load_runner()
    from sdm_robustness.pipeline.core import clean_predictors, get_track_columns
    from trustworthy_sdm.analysis import ENTITY_NAME_TO_DIR
    from trustworthy_sdm.regen import assemble_inputs

    with open(REPO / "data/protocol_b_predictor_sets.json") as f:
        pred_sets = json.load(f)
    rows = []
    t0 = time.time()
    for entity in rpb.ENTITIES:
        edir = ENTITY_NAME_TO_DIR[entity]
        inputs = None
        for track in rpb.TRACKS:
            bench = load_members(SURF / f"{edir}__consensus__{track}__benchmark__L0").mean(axis=1)
            for level in rpb.LEVELS:
                cell = f"{edir}__consensus__{track}__lowacc__L{level}"
                ndir = NULL / cell
                if not (ndir / "rep_03.parquet").exists():
                    if inputs is None:
                        inputs = assemble_inputs(entity=entity, track="combined", axis=rpb.AXIS,
                                                 master_csv=MASTER)
                    full_cols = clean_predictors(inputs.benchmark, get_track_columns(inputs.benchmark, track),
                                                 missing_threshold_pct=rpb.MISSING_THRESHOLD_PCT)
                    reduced_cols = pred_sets[f"{entity}|||{track}"]["predictors"]
                    pres = null_presences(inputs.benchmark, level, rpb.SEED)
                    rpb.write_cell(fit_consensus(rpb, inputs, pres, reduced_cols, full_cols), ndir)
                for source, root in (("contaminated", SURF), ("null", NULL)):
                    for frame in directional(load_members(root / cell), bench):
                        rows.append(frame.assign(entity=entity, entity_dir=edir, track=track,
                                                 level=level, source=source))
            print(f"{entity} | {track}: done ({time.time() - t0:.0f}s)", flush=True)
    res = pd.concat(rows, ignore_index=True)

    sub = pd.concat([pd.read_csv(REPO / f"figures/asymmetry_protocol_b_5bin_L{lv}.csv") for lv in (10, 20)],
                    ignore_index=True)
    sub["bin"] = sub.groupby(["entity", "track", "level"])["bench_decile_mid"].rank(method="first").astype(int) - 1
    mine = res[(res["source"] == "contaminated") & (res["binning"] == "benchmark")]
    chk = mine.merge(sub[["entity", "track", "level", "bin", "frac_over_predicted"]],
                     on=["entity", "track", "level", "bin"])
    rep_err = float((chk["frac_over"] - chk["frac_over_predicted"]).abs().max()) if len(chk) else float("nan")
    print(f"\nreproduction of submitted Fig. 2 values: {len(chk)} of {len(sub)} bins matched, max |diff| {rep_err:.2e}")

    bb = res[res["binning"] == "benchmark"].copy()
    keys = ["entity_dir", "track", "level", "source"]
    bb["last"] = bb.groupby(keys)["bin"].transform("max")
    first = bb[bb["bin"] == 0].set_index(keys)["frac_over"]
    last = bb[bb["bin"] == bb["last"]].set_index(keys)["frac_over"]
    grad = (first - last).rename("gradient").reset_index()
    gt = grad.pivot_table(index=["track", "level"], columns="source", values="gradient", aggfunc="mean")
    gt["excess"] = gt["contaminated"] - gt["null"]
    print("\ngradient (lowest minus highest bin, fraction over-predicted), panel mean:")
    print(gt.round(3).to_string())

    sc = grad[grad["track"].isin(SCORED)].pivot_table(index="level", columns="source",
                                                      values="gradient", aggfunc="mean")
    sc["excess"] = sc["contaminated"] - sc["null"]
    sc["null_share"] = sc["null"] / sc["contaminated"]
    print("\nscored tracks (local_only + combined):")
    print(sc.round(3).to_string())
    ex = sc["excess"].sort_index().to_numpy()
    c1 = bool((ex > 0).all())
    c2 = bool(np.all(np.diff(ex) > 0))
    print(f"criterion 1 (excess > 0 at every level): {'PASS' if c1 else 'FAIL'}")
    print(f"criterion 2 (excess increases with level): {'PASS' if c2 else 'FAIL'}")

    md = bb[bb["track"].isin(SCORED)].pivot_table(index=["level", "source"], columns="bin",
                                                  values="mean_diff", aggfunc="mean")
    print("\nmean signed divergence (consensus minus benchmark) by benchmark bin, scored tracks:")
    print(md.round(4).to_string())
    ba = res[(res["binning"] == "bland_altman") & res["track"].isin(SCORED)].pivot_table(
        index=["level", "source"], columns="bin", values="frac_over", aggfunc="mean")
    print("\nfraction over-predicted by Bland-Altman bin (mean of the two predictions), scored tracks:")
    print(ba.round(3).to_string())

    OUT_BINS.parent.mkdir(parents=True, exist_ok=True)
    res.round(5).to_csv(OUT_BINS, index=False)
    gt.round(5).to_csv(OUT_SUMM)
    sys.exit(0 if (len(chk) > 0 and rep_err < 1e-9) else 1)


if __name__ == "__main__":
    main()
