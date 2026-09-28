"""EDS-2026-0083 revision: environment agreement for the Protocol A null comparison.

The Protocol A null replicates (protocol_a_null_panel.csv) were refitted on this
workstation; the submitted contaminated replicates were generated in the May 2026
environment (VEGA), which this environment reproduces to r = 0.998-0.999 per surface
(so far verified for the worked case only, where coverage agreed within 0.002). So that
the null-adjusted counts never compare arms from different environments, the 30
contaminated replicates of every competent cell are regenerated here with the same code
(rev_eds_worked_case_null.refit_surface / contaminated_builder; worked-case cells reused)
and their coverage of the deterministic benchmark is compared with the submitted one.

Pre-registered (fixed and committed before the run):
  agreement  |local - submitted coverage| <= 0.02 in every cell;
  if agreement holds, the null-adjusted counts stay as computed from the submitted arm;
  if it fails in any cell, the null-adjusted counts are recomputed with the local
  contaminated arm for all cells (same environment as the null), and those replace them.
Output: reports/rev_eds/protocol_a_local_agreement.csv.
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
LOCAL_ROOT = REPO / "data/rev_eds/protocol_a_local"
PANEL = REPO / "reports/rev_eds/protocol_a_null_panel.csv"
OUT = REPO / "reports/rev_eds/protocol_a_local_agreement.csv"
LEVELS = (3, 10, 20)
GATE, TOL = 0.70, 0.02
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

    panel = pd.read_csv(PANEL)
    g = pd.read_parquet(REPO / "data/results/grid_b_merged/grid_b_results_raw_merged.parquet",
                        columns=["entity", "algorithm", "track", "axis", "level", "replicate", "seed"])
    g = g[(g["axis"] == "lowacc") & g["level"].isin(LEVELS)].copy()
    g["level"] = g["level"].astype(int)
    g["entity_dir"] = g["entity"].map(ENTITY_NAME_TO_DIR)
    g["algorithm_n"] = g["algorithm"].astype(str).str.lower().map(ALGO).fillna(g["algorithm"])
    g["track_n"] = g["track"].astype(str).map(TRACK).fillna(g["track"])
    cells = (g[["entity", "entity_dir", "algorithm", "algorithm_n", "track", "track_n", "level"]].drop_duplicates()
             .merge(panel[CKEYS].rename(columns={"algorithm": "algorithm_n", "track": "track_n"}),
                    on=["entity_dir", "algorithm_n", "track_n", "level"]))
    if len(cells) != len(panel):
        sys.exit(f"matched {len(cells)} of {len(panel)} panel cells")
    seeds = {k: dict(zip(d["replicate"].astype(int), d["seed"].astype(int)))
             for k, d in g.groupby(["entity", "algorithm", "track", "level"])}

    def cid(entity, algorithm, track, level):
        return CellID(entity, algorithm, track, axis="lowacc", level=int(level))

    todo = []
    for c in cells.itertuples(index=False):
        d = LOCAL_ROOT / cid(c.entity, c.algorithm, c.track, c.level).short()
        for rep, seed in sorted(seeds[(c.entity, c.algorithm, c.track, c.level)].items()):
            if not (d / f"rep_{rep:02d}.parquet").exists():
                todo.append((c.entity, c.track, c.algorithm, c.level, rep, seed))
    print(f"{len(cells)} cells; {len(todo)} local contaminated fits to run", flush=True)
    t0, done = time.time(), 0
    todo = pd.DataFrame(todo, columns=["entity", "track", "algorithm", "level", "rep", "seed"])
    for (entity, track), grp in todo.groupby(["entity", "track"], sort=True):
        inputs = assemble_inputs(entity=entity, track=track, axis="lowacc", master_csv=MASTER)
        for r in grp.itertuples(index=False):
            out = LOCAL_ROOT / cid(entity, r.algorithm, track, r.level).short()
            out.mkdir(parents=True, exist_ok=True)
            tmp = out / f".rep_{int(r.rep):02d}.tmp.parquet"
            wc.refit_surface(inputs, track, r.algorithm, int(r.seed),
                             wc.contaminated_builder(int(r.level))).to_parquet(tmp, index=False)
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
        cell = cid(c.entity, c.algorithm, c.track, c.level)
        ens = load_ensemble(cell, LOCAL_ROOT)
        idx = ens.index.intersection(bench.index)
        m = ens.loc[idx].to_numpy()
        b = bench.loc[idx].to_numpy()
        lo, hi = np.percentile(m, [2.5, 97.5], axis=1)
        r0 = min(seeds[(c.entity, c.algorithm, c.track, c.level)])
        a = pd.read_parquet(LOCAL_ROOT / cell.short() / f"rep_{r0:02d}.parquet")
        s = pd.read_parquet(ci.SURFACES_ROOT / cell.short() / f"rep_{r0:02d}.parquet")
        a = a.set_index(a["subc_id"].astype(str))["predicted_probability"]
        s = s.set_index(s["subc_id"].astype(str))["predicted_probability"]
        k = a.index.intersection(s.index)
        rows.append({"entity_dir": c.entity_dir, "algorithm": c.algorithm_n, "track": c.track_n,
                     "level": int(c.level), "local_coverage": float(np.mean((b >= lo) & (b <= hi))),
                     "local_width": float(np.mean(hi - lo)),
                     "rep0_max_abs_diff": float((a.loc[k] - s.loc[k]).abs().max()),
                     "rep0_r": float(np.corrcoef(a.loc[k], s.loc[k])[0, 1])})
    res = panel.merge(pd.DataFrame(rows), on=CKEYS)
    res["local_minus_submitted"] = res["local_coverage"] - res["coverage_uncorrected"]
    gate = res["auc_cont_oos"] >= GATE
    res["miscovered_adj_local"] = res["local_coverage"] < res["threshold"]
    res["silent_adj_local"] = gate & res["miscovered_adj_local"]
    res.sort_values(CKEYS).round(5).to_csv(OUT, index=False)

    d = res["local_minus_submitted"].abs()
    worst = res.loc[d.idxmax()]
    agree = bool((d <= TOL).all())
    print(f"\ncoverage, local - submitted: median |diff| {d.median():.4f}, max {d.max():.4f} "
          f"({worst['entity_dir']}, {worst['algorithm']}, {worst['track']}, L{int(worst['level'])}); "
          f"cells beyond {TOL}: {int((d > TOL).sum())}/{len(res)}")
    print(f"replicate-0 surface, local vs submitted: r {res['rep0_r'].min():.4f}-{res['rep0_r'].max():.4f}, "
          f"max |diff| up to {res['rep0_max_abs_diff'].max():.3f}")
    cols = ["entity_dir", "algorithm", "track", "level", "null_coverage", "null_width", "local_width",
            "coverage_uncorrected", "local_coverage"]
    print("\nlowest null coverage:")
    print(res.nsmallest(6, "null_coverage")[cols].round(3).to_string(index=False))
    print("\nsilent/miscovered, null-adjusted: submitted arm | local arm")
    for lvl in LEVELS:
        r = res[res["level"] == lvl]
        print(f"  L{lvl:>2}: {int(r['silent_adj'].sum())}/{int(r['miscovered_adj'].sum())} | "
              f"{int(r['silent_adj_local'].sum())}/{int(r['miscovered_adj_local'].sum())}")
    print(f"\npre-registered reading: agreement {'HOLDS' if agree else 'FAILS'} -> null-adjusted counts from the "
          f"{'submitted' if agree else 'local'} arm")


if __name__ == "__main__":
    main()
