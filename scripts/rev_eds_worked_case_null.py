"""EDS-2026-0083 revision: perturbation null for the worked case (P. leniusculus, Protocol A).

The worked case (section 3.7, figure 5b) reports the divergence of 30 Random
Forest replicate surfaces (combined track, non-native range) from the
companion study's deterministic benchmark, by fixed benchmark-suitability
band, with 95% intervals across replicates. It was computed without a null.

The submitted replicate surfaces were generated on the VEGA cluster; on this
workstation the pinned code reproduces them only to r = 0.998 (differences
between software environments), so both arms are regenerated here with
identical code and environment:
  contaminated  each submitted replicate's own recipe (logged seed; same
                retained presences, background draw, model randomness,
                predictor screen and training size);
  null          the same recipe with the replaced records drawn as duplicates
                of retained high-accuracy records instead of low-accuracy ones.
The final refit of sdm-robustness v1.0 fit_cv_cell is replicated; a code gate
requires it to match the repository's own regenerate_cell bit for bit.

Band statistics use run_overpred_ci.py's own bands and benchmark, and the
submitted band means are reproduced first. Excess = contaminated minus null,
paired by replicate seed; intervals are 2.5-97.5 percentiles over the 30
replicates.

Fixed before the run:
  agreement  the local regeneration reproduces the submitted worked case if
             every regenerated band mean lies inside the submitted 95%
             replicate interval; otherwise the text reports the regenerated
             values, and says so;
  (1) the core-band deflation (benchmark 0.7-1.0) is retained only if the
      paired excess is negative with its 95% interval entirely below zero at
      L3, L10 and L20;
  (2) the edge-band inflation (0.1-0.3) is confirmed if the paired excess is
      positive with its interval entirely above zero at all three levels and
      its mean increases from L3 to L20.

Output: reports/rev_eds/worked_case_null_bands.csv
Surfaces (record-level) go to data/rev_eds/protocol_a_local/ and
data/rev_eds/protocol_a_null/ (local only).
"""
import importlib.util
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
MASTER = Path(os.environ.get("TS_MASTER_CSV") or
              REPO.parent / "sdm-robustness/data/raw/combined_data_true_master.csv")
LOCAL_ROOT = REPO / "data/rev_eds/protocol_a_local"
NULL_ROOT = REPO / "data/rev_eds/protocol_a_null"
OUT = REPO / "reports/rev_eds/worked_case_null_bands.csv"
LEVELS = (3, 10, 20)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def contaminated_builder(level):
    from sdm_robustness.pipeline.core import contaminate_presence_set
    return lambda bench, pool, seed: contaminate_presence_set(
        benchmark=bench, contamination_pool=pool, level_pct=level, seed=seed)


def null_builder(level):
    def build(bench, pool, seed):
        n_replace = int(round(len(bench) * level / 100.0))
        kept = bench.sample(n=len(bench) - n_replace, replace=False, random_state=seed)
        dup = kept.sample(n=n_replace, replace=True, random_state=seed + 1)
        return pd.concat([kept, dup], ignore_index=True)
    return build


def refit_surface(inputs, track, algorithm, seed, pres_builder):
    """Final refit of sdm-robustness v1.0 fit_cv_cell, with a pluggable presence set."""
    from sdm_robustness.pipeline.core import (build_model, clean_predictors, get_track_columns,
                                              predict_suitability_surface)
    bench = inputs.benchmark
    kept = clean_predictors(bench, get_track_columns(bench, track))
    med = bench[kept].median(numeric_only=True)
    pres = pres_builder(bench, inputs.contamination_pool, seed)[["subc_id", "basin_id"] + kept].copy()
    pres[kept] = pres[kept].fillna(med)
    acc = inputs.accessible_area[["subc_id", "basin_id"] + kept].copy()
    acc[kept] = acc[kept].fillna(med)
    n_neg = min(int(round(len(pres) * 1.0)), len(acc))
    neg = acc.sample(n=n_neg, replace=False, random_state=seed)
    x = pd.concat([pres[kept], neg[kept]], axis=0)
    y = np.array([1] * len(pres) + [0] * n_neg)
    model = build_model(algorithm, seed=seed, n_jobs=-1)
    model.fit(x, y)
    surf = predict_suitability_surface(model, acc[kept])
    return pd.DataFrame({"subc_id": inputs.accessible_area["subc_id"].values,
                         "predicted_probability": np.asarray(surf, dtype=float)})


def compare(a, b):
    a = a.set_index(a["subc_id"].astype(str))["predicted_probability"]
    b = b.set_index(b["subc_id"].astype(str))["predicted_probability"]
    c = a.index.intersection(b.index)
    return float((a.loc[c] - b.loc[c]).abs().max()), float(np.corrcoef(a.loc[c], b.loc[c])[0, 1])


def band_reps(ens, bench, band, cols):
    b = bench.to_numpy()
    out = {}
    for k, bnd in enumerate(band.cat.categories):
        m = (band == bnd).to_numpy()
        out[k] = np.array([(ens[col].to_numpy()[m] - b[m]).mean() for col in cols])
    return out


def fmt(m, lo, hi):
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def main():
    warnings.simplefilter("ignore")
    ci = load_module("run_overpred_ci", REPO / "run_overpred_ci.py")
    from trustworthy_sdm.analysis import benchmark_for
    from trustworthy_sdm.conformal import CellID, load_ensemble
    from trustworthy_sdm.regen import assemble_inputs, regenerate_cell

    entity, algo, track = ci.ENTITY, ci.ALGO, ci.TRACK
    inputs = assemble_inputs(entity=entity, track=track, axis="lowacc", master_csv=MASTER)
    g = pd.read_parquet(REPO / "data/results/grid_b_merged/grid_b_results_raw_merged.parquet",
                        columns=["entity", "algorithm", "track", "axis", "level", "replicate", "seed"])
    seeds = {}
    for level in LEVELS:
        s = g[(g["entity"] == entity) & (g["algorithm"] == algo) & (g["track"] == track)
              & (g["axis"] == "lowacc") & (g["level"] == level)]
        seeds[level] = dict(zip(s["replicate"].astype(int), s["seed"].astype(int)))

    cell3 = CellID(entity, algo, track, axis="lowacc", level=3)
    r0 = min(seeds[3])
    tmp = Path(tempfile.mkdtemp())
    regenerate_cell(cell3, inputs, pd.DataFrame({"replicate": [r0], "seed": [seeds[3][r0]]}), tmp)
    d, _ = compare(refit_surface(inputs, track, algo, seeds[3][r0], contaminated_builder(3)),
                   pd.read_parquet(tmp / cell3.short() / f"rep_{r0:02d}.parquet"))
    print(f"code gate (this refit vs repository regenerate_cell): max |diff| {d:.2e} -> {'PASS' if d < 1e-9 else 'FAIL'}")
    if d >= 1e-9:
        sys.exit(1)

    t0 = time.time()
    for level in LEVELS:
        cell = CellID(entity, algo, track, axis="lowacc", level=level)
        for root, builder in ((LOCAL_ROOT, contaminated_builder(level)), (NULL_ROOT, null_builder(level))):
            out_dir = root / cell.short()
            out_dir.mkdir(parents=True, exist_ok=True)
            for rep, seed in sorted(seeds[level].items()):
                f = out_dir / f"rep_{rep:02d}.parquet"
                if not f.exists():
                    refit_surface(inputs, track, algo, seed, builder).to_parquet(f, index=False)
        rr = min(seeds[level])
        dd, cc = compare(pd.read_parquet(LOCAL_ROOT / cell.short() / f"rep_{rr:02d}.parquet"),
                         pd.read_parquet(ci.SURFACES_ROOT / cell.short() / f"rep_{rr:02d}.parquet"))
        print(f"L{level}: 30 local + 30 null replicates ({time.time() - t0:.0f}s); "
              f"replicate {rr} local vs submitted: max |diff| {dd:.2e}, r = {cc:.4f}", flush=True)

    bench = benchmark_for(entity, algo, track, ci.PATHS)
    rows, rep_err = [], 0.0
    for level in LEVELS:
        cell = CellID(entity, algo, track, axis="lowacc", level=level)
        ens = {name: load_ensemble(cell, root) for name, root in
               (("submitted", ci.SURFACES_ROOT), ("local", LOCAL_ROOT), ("null", NULL_ROOT))}
        cols = [c for c in ens["local"].columns if c in ens["null"].columns and c in ens["submitted"].columns]
        shared = bench.index
        for e in ens.values():
            shared = shared.intersection(e.index)
        b = bench.loc[shared]
        band = pd.cut(b, ci.BANDS, right=False, include_lowest=True)
        reps = {name: band_reps(e.loc[shared], b, band, cols) for name, e in ens.items()}
        ref = ci.analyse_level_ci(level)
        for k, bnd in enumerate(band.cat.categories):
            sub, loc, nul = reps["submitted"][k], reps["local"][k], reps["null"][k]
            exc = loc - nul
            rep_err = max(rep_err, abs(round(float(sub.mean()), 4) - ref[str(bnd)]["cellmean_div"]["mean"]))
            p = lambda v: np.percentile(v, [2.5, 97.5])
            rows.append({"level": level, "band_index": k, "band": str(bnd), "n_subc": int((band == bnd).sum()),
                         "n_reps": len(cols),
                         "sub_mean": sub.mean(), "sub_lo": p(sub)[0], "sub_hi": p(sub)[1],
                         "local_mean": loc.mean(), "local_lo": p(loc)[0], "local_hi": p(loc)[1],
                         "null_mean": nul.mean(), "null_lo": p(nul)[0], "null_hi": p(nul)[1],
                         "excess_mean": exc.mean(), "excess_lo": p(exc)[0], "excess_hi": p(exc)[1]})
    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.round(5).to_csv(OUT, index=False)
    print(f"\nreproduction of the submitted band means (run_overpred_ci.py): max |diff| {rep_err:.1e}")
    show = pd.DataFrame({
        "level": res["level"], "band": res["band"], "n": res["n_subc"],
        "submitted": [fmt(r.sub_mean, r.sub_lo, r.sub_hi) for r in res.itertuples()],
        "local": [fmt(r.local_mean, r.local_lo, r.local_hi) for r in res.itertuples()],
        "null": [fmt(r.null_mean, r.null_lo, r.null_hi) for r in res.itertuples()],
        "excess": [fmt(r.excess_mean, r.excess_lo, r.excess_hi) for r in res.itertuples()]})
    print(show.to_string(index=False))

    agree = bool(((res["local_mean"] >= res["sub_lo"]) & (res["local_mean"] <= res["sub_hi"])).all())
    last = res["band_index"].max()
    core = res[res["band_index"] == last].sort_values("level")
    edge = res[res["band_index"] == 1].sort_values("level")
    c1 = bool((core["excess_hi"] < 0).all())
    c2 = bool((edge["excess_lo"] > 0).all() and np.all(np.diff(edge["excess_mean"].to_numpy()) > 0))
    print(f"\nagreement (every local band mean inside the submitted 95% interval): {'PASS' if agree else 'FAIL'}")
    print(f"criterion 1 (core deflation exceeds the null at every level): {'PASS' if c1 else 'FAIL'}")
    print(f"criterion 2 (edge inflation exceeds the null and rises with level): {'PASS' if c2 else 'FAIL'}")

    try:
        ro = load_module("run_overprediction", REPO / "run_overprediction.py")
        for level in LEVELS:
            out = {}
            for name, root in (("local", LOCAL_ROOT), ("null", NULL_ROOT)):
                ro.SURFACES_ROOT = root
                out[name] = ro.analyse_level(level, 0.5)[3]
            print(f"flips at tau 0.5, L{level}: local {out['local']} | null {out['null']}")
    except Exception as e:  # noqa: BLE001
        print(f"flip comparison skipped: {type(e).__name__}: {e}")
    sys.exit(0 if rep_err < 1e-4 else 1)


if __name__ == "__main__":
    main()
