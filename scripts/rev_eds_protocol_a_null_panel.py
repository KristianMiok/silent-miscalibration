"""EDS-2026-0083 revision: panel-wide null-refit coverage for Protocol A.

The worked case (worked_case_null_coverage.csv) shows that Protocol A's 30-replicate
interval covers the companion study's deterministic benchmark at only 0.88-0.905 without
contamination (matched-perturbation null), against 0.98-1.00 for Protocol B
(null_coverage.csv). The miscoverage threshold behind the silent-failure counts
(0.90 = 0.95 x 0.95) assumes the benchmark's own coverage is 0.95; for Protocol A that
assumption fails in the worked case.

For every competent Protocol A cell (clean basin-blocked CV AUC >= 0.70; 40
configurations x L3/L10/L20), 30 null replicates are refitted with each submitted
replicate's own recipe (logged Grid B seed; same retained presences, background,
predictor screen and training size), the replaced records drawn as duplicates of
retained high-accuracy records (rev_eds_worked_case_null.refit_surface / null_builder,
code-gated there against the repository's regenerate_cell). Null coverage c0 = coverage
of the deterministic benchmark by the null 2.5-97.5 % replicate interval. Surfaces are
cached in data/rev_eds/protocol_a_null (worked-case cells reused); the run is resumable.

Pre-registered rule (fixed and committed before the run):
  threshold(cell) = min(0.90, 0.95 x c0(cell)): the assumed benchmark coverage 0.95 in
  0.95^2 is replaced by the measured one where it is lower, never where it is higher
  (Protocol B's c0 >= 0.978 leaves its threshold at 0.90 and its counts unchanged);
  miscovered = coverage_uncorrected < threshold; silent = competent, passes the 0.70
  gate after contamination, and miscovered. These null-adjusted counts replace the
  0.90 counts for Protocol A in the text, whatever they are.
Output: reports/rev_eds/protocol_a_null_panel.csv (one row per competent cell).
"""
import importlib.util
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
os.environ.setdefault("TS_MASTER_CSV", str(MASTER))
NULL_ROOT = REPO / "data/rev_eds/protocol_a_null"
JOIN = REPO / "reports/rev_eds/item1_protocol_a_join.csv"
WCC = REPO / "reports/rev_eds/worked_case_null_coverage.csv"
OUT = REPO / "reports/rev_eds/protocol_a_null_panel.csv"
LEVELS = (3, 10, 20)
NOMINAL, THRESH, GATE = 0.95, 0.90, 0.70
ALGO = {"rf": "random_forest", "random_forest": "random_forest", "xgb": "xgboost", "xgboost": "xgboost"}
TRACK = {"local": "local_only", "local_only": "local_only", "upstream": "upstream_only",
         "upstream_only": "upstream_only", "combined": "combined"}
CKEYS = ["entity_dir", "algorithm", "track", "level"]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    warnings.simplefilter("ignore")
    wc = load_module("wc", REPO / "scripts/rev_eds_worked_case_null.py")
    ci = load_module("run_overpred_ci", REPO / "run_overpred_ci.py")
    from trustworthy_sdm.analysis import ENTITY_NAME_TO_DIR, benchmark_for
    from trustworthy_sdm.conformal import CellID, load_ensemble
    from trustworthy_sdm.regen import assemble_inputs

    j = pd.read_csv(JOIN)
    for col in ("competent", "passes_gate_contaminated"):
        j[col] = j[col].astype(str).str.lower().eq("true")
    comp = (j[j["competent"]][["entity_dir", "algorithm", "track"]].drop_duplicates()
            .rename(columns={"algorithm": "algorithm_n", "track": "track_n"}))

    g = pd.read_parquet(REPO / "data/results/grid_b_merged/grid_b_results_raw_merged.parquet",
                        columns=["entity", "algorithm", "track", "axis", "level", "replicate", "seed"])
    g = g[(g["axis"] == "lowacc") & g["level"].isin(LEVELS)].copy()
    g["level"] = g["level"].astype(int)
    g["entity_dir"] = g["entity"].map(ENTITY_NAME_TO_DIR)
    g["algorithm_n"] = g["algorithm"].astype(str).str.lower().map(ALGO).fillna(g["algorithm"])
    g["track_n"] = g["track"].astype(str).map(TRACK).fillna(g["track"])
    cells = (g[["entity", "entity_dir", "algorithm", "algorithm_n", "track", "track_n", "level"]]
             .drop_duplicates().merge(comp, on=["entity_dir", "algorithm_n", "track_n"]))
    print(f"{len(comp)} competent configurations -> {len(cells)} cells")
    if len(cells) != 3 * len(comp):
        sys.exit(f"expected {3 * len(comp)} cells; check entity/algorithm/track naming")
    seeds = {k: dict(zip(d["replicate"].astype(int), d["seed"].astype(int)))
             for k, d in g.groupby(["entity", "algorithm", "track", "level"])}

    def cid(entity, algorithm, track, level):
        return CellID(entity, algorithm, track, axis="lowacc", level=int(level))

    todo = []
    for c in cells.itertuples(index=False):
        d = NULL_ROOT / cid(c.entity, c.algorithm, c.track, c.level).short()
        for rep, seed in sorted(seeds[(c.entity, c.algorithm, c.track, c.level)].items()):
            if not (d / f"rep_{rep:02d}.parquet").exists():
                todo.append((c.entity, c.track, c.algorithm, c.level, rep, seed))
    total = sum(len(seeds[(c.entity, c.algorithm, c.track, c.level)]) for c in cells.itertuples(index=False))
    print(f"{len(todo)} null fits to run, {total - len(todo)} cached", flush=True)

    t0, done = time.time(), 0
    todo = pd.DataFrame(todo, columns=["entity", "track", "algorithm", "level", "rep", "seed"])
    for (entity, track), grp in todo.groupby(["entity", "track"], sort=True):
        inputs = assemble_inputs(entity=entity, track=track, axis="lowacc", master_csv=MASTER)
        for r in grp.itertuples(index=False):
            out = NULL_ROOT / cid(entity, r.algorithm, track, r.level).short()
            out.mkdir(parents=True, exist_ok=True)
            tmp = out / f".rep_{int(r.rep):02d}.tmp.parquet"
            wc.refit_surface(inputs, track, r.algorithm, int(r.seed), wc.null_builder(int(r.level))).to_parquet(
                tmp, index=False)
            tmp.rename(out / f"rep_{int(r.rep):02d}.parquet")
            done += 1
        el = time.time() - t0
        print(f"  {entity} | {track}: {done}/{len(todo)} fits, {el / 60:.0f} min elapsed, "
              f"~{el / done * (len(todo) - done) / 60:.0f} min left", flush=True)

    rows, bench_cache = [], {}
    for c in cells.itertuples(index=False):
        key = (c.entity, c.algorithm, c.track)
        if key not in bench_cache:
            bench_cache[key] = benchmark_for(c.entity, c.algorithm, c.track, ci.PATHS)
        bench = bench_cache[key]
        ens = load_ensemble(cid(c.entity, c.algorithm, c.track, c.level), NULL_ROOT)
        idx = ens.index.intersection(bench.index)
        m = ens.loc[idx].to_numpy()
        b = bench.loc[idx].to_numpy()
        lo, hi = np.percentile(m, [2.5, 97.5], axis=1)
        rows.append({"entity_dir": c.entity_dir, "algorithm": c.algorithm_n, "track": c.track_n,
                     "level": int(c.level), "n_reps": m.shape[1], "n_subc": len(idx),
                     "null_coverage": float(np.mean((b >= lo) & (b <= hi))),
                     "null_width": float(np.mean(hi - lo))})
    res = pd.DataFrame(rows).merge(
        j[CKEYS + ["coverage_uncorrected", "auc_cont_oos", "passes_gate_contaminated"]], on=CKEYS, how="left")
    res["threshold"] = np.minimum(THRESH, NOMINAL * res["null_coverage"])
    res["deficit_vs_null"] = res["null_coverage"] - res["coverage_uncorrected"]
    gate = res["auc_cont_oos"] >= GATE
    res["miscovered_090"] = res["coverage_uncorrected"] < THRESH
    res["miscovered_adj"] = res["coverage_uncorrected"] < res["threshold"]
    res["silent_090"] = gate & res["miscovered_090"]
    res["silent_adj"] = gate & res["miscovered_adj"]
    res.sort_values(CKEYS).round(5).to_csv(OUT, index=False)

    wcc = pd.read_csv(WCC, keep_default_na=False, na_values=[""])
    wn = wcc[wcc["ensemble"] == "null_refit"].set_index("level")["coverage"]
    me = res[(res["entity_dir"] == ENTITY_NAME_TO_DIR[ci.ENTITY]) & (res["algorithm"] == "random_forest")
             & (res["track"] == "combined")].set_index("level")["null_coverage"]
    print(f"\nworked-case cells vs worked_case_null_coverage.csv: max |diff| {(me - wn).abs().max():.1e}")
    print("by level, competent cells (0.90-rule counts should reproduce 26/26, 34/34, 36/40):")
    for lvl in LEVELS:
        r = res[res["level"] == lvl]
        print(f"  L{lvl:>2}: null coverage median {r['null_coverage'].median():.3f} "
              f"[{r['null_coverage'].min():.3f}-{r['null_coverage'].max():.3f}, below 0.90 in "
              f"{int((r['null_coverage'] < THRESH).sum())}/{len(r)}]; contaminated median "
              f"{r['coverage_uncorrected'].median():.3f}; deficit vs null median {r['deficit_vs_null'].median():+.3f}; "
              f"silent/miscovered 0.90 rule {int(r['silent_090'].sum())}/{int(r['miscovered_090'].sum())}, "
              f"null-adjusted {int(r['silent_adj'].sum())}/{int(r['miscovered_adj'].sum())}")
    print("\nnull coverage, median by algorithm and track:")
    print(res.pivot_table(index="algorithm", columns="track", values="null_coverage", aggfunc="median")
          .round(3).to_string())


if __name__ == "__main__":
    main()
